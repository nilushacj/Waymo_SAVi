# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main model training loop with gradient accumulation."""

import functools
import os
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Type, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from savi.lib import evaluator
from savi.lib import input_pipeline
from savi.lib import losses
from savi.lib import utils
import tensorflow as tf

Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]
PRNGKey = Array

#############################
# New train_step_accum function
#############################
def train_step_accum(
    model: nn.Module,
    rng: PRNGKey,
    step: int,
    state_vars: flax.core.FrozenDict,
    opt: flax.optim.Optimizer,  # pytype: disable=module-attr
    batch: Dict[str, ArrayTree],
    loss_fn: losses.LossFn,
    learning_rate_fn: Callable[[Array], Array],
    train_metrics_cls: Type[metrics.Collection],
    predicted_max_num_instances: int,
    ground_truth_max_num_instances: int,
    conditioning_key: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
    accum_grad: Optional[Any] = None,
    accum_steps: int = 0,
    accumulation_steps: int = 4,
) -> Tuple[
    flax.optim.Optimizer,
    flax.core.FrozenDict,
    PRNGKey,
    metrics.Collection,
    int,
    Optional[Any],
    int,
]:
    """Perform a single training step with gradient accumulation.

    Accumulates gradients over `accumulation_steps` mini-batches before updating
    model parameters.
    """
    # --- RNG handling ---
    new_rng, rng = jax.random.split(rng)
    rng = jax.random.fold_in(rng, jax.host_id())
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    init_rng, dropout_rng = jax.random.split(rng, 2)

    mutable_var_keys = list(state_vars.keys()) + ["intermediates"]
    conditioning = batch[conditioning_key] if conditioning_key else None

    def train_loss_fn(params, state_vars):
        preds, mutable_vars = model.apply(
            {"params": params, **state_vars},
            video=batch["video"],
            conditioning=conditioning,
            mutable=mutable_var_keys,
            rngs={"state_init": init_rng, "dropout": dropout_rng},
            train=True,
            padding_mask=batch.get("padding_mask"),
        )
        # Remove intermediates from state_vars.
        state_vars = utils.filter_key_from_frozen_dict(mutable_vars, key="intermediates")
        loss, loss_aux = loss_fn(preds, batch)
        # Scale loss down by accumulation_steps.
        return loss / accumulation_steps, (state_vars, preds, loss_aux)

    grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
    (loss, (state_vars, preds, loss_aux)), grad = grad_fn(opt.target, state_vars)

    # Average gradients over devices.
    grad = jax.lax.pmean(grad, axis_name="batch")

    if max_grad_norm is not None:
        grad = utils.clip_grads(grad, max_grad_norm)

    # Accumulate gradients.
    if accum_grad is None:
        accum_grad = grad
    else:
        accum_grad = jax.tree_map(lambda a, b: a + b, accum_grad, grad)
    accum_steps += 1

    # Update parameters if enough mini-batches have been accumulated.
    if accum_steps == accumulation_steps:
        learning_rate = learning_rate_fn(step - 1)
        opt = opt.apply_gradient(accum_grad, learning_rate=learning_rate)
        accum_grad = None
        accum_steps = 0

    # Compute training metrics (scale the loss back for reporting).
    metrics_update = train_metrics_cls.gather_from_model_output(
        loss=loss * accumulation_steps,
        **loss_aux,
        predicted_segmentations=utils.remove_singleton_dim(
            preds["outputs"].get("segmentations")),
        ground_truth_segmentations=utils.remove_singleton_dim(
            preds["outputs"].get("segmentations")),
        predicted_max_num_instances=predicted_max_num_instances,
        ground_truth_max_num_instances=ground_truth_max_num_instances,
        padding_mask=batch.get("padding_mask"),
        mask=batch.get("mask"))
    return opt, state_vars, new_rng, metrics_update, step + 1, accum_grad, accum_steps

#############################
# End of train_step_accum function
#############################

def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str):
  """Runs a training and evaluation loop with gradient accumulation."""
  # ---- a key to control randomness ----
  rng = jax.random.PRNGKey(config.seed)

  # ---- make the work directory + sub directories----
  tf.io.gfile.makedirs(workdir)

  # Input pipeline.
  rng, data_rng = jax.random.split(rng)
  if config.get("seed_data", True):
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
  else:
    data_rng = None

  train_ds, eval_ds = input_pipeline.create_datasets(config, data_rng)
  train_iter = iter(train_ds)

  # Initialize model.
  model = utils.build_model_from_config(config.model)
  learning_rate_fn = optax.warmup_cosine_decay_schedule(
      init_value=0.,
      peak_value=config.learning_rate,
      warmup_steps=config.warmup_steps,
      decay_steps=config.num_train_steps)
  optimizer_def = flax.optim.Adam(learning_rate=config.learning_rate)

  train_metrics_cls = utils.make_metrics_collection("TrainMetrics",
                                                    config.train_metrics_spec)
  eval_metrics_cls = utils.make_metrics_collection("EvalMetrics",
                                                   config.eval_metrics_spec)

  def init_model(rng):
    rng, init_rng, model_rng, dropout_rng = jax.random.split(rng, num=4)
    init_conditioning = None
    if config.get("conditioning_key"):
      init_conditioning = jnp.ones(
          [1] + list(train_ds.element_spec[config.conditioning_key].shape)[2:],
          jnp.int32)
    init_inputs = jnp.ones(
        [1] + list(train_ds.element_spec["video"].shape)[2:],
        jnp.float32)
    initial_vars = model.init(
        {"params": model_rng, "state_init": init_rng, "dropout": dropout_rng},
        video=init_inputs, conditioning=init_conditioning,
        padding_mask=jnp.ones(init_inputs.shape[:-1], jnp.int32))
    state_vars, initial_params = initial_vars.pop("params")
    state_vars = utils.filter_key_from_frozen_dict(state_vars, key="intermediates")
    return state_vars, initial_params

  state_vars, initial_params = init_model(rng)
  parameter_overview.log_parameter_overview(initial_params)
  optimizer = optimizer_def.create(initial_params)
  state = utils.TrainState(
      step=1, optimizer=optimizer, rng=rng, variables=state_vars)

  loss_fn = functools.partial(
      losses.compute_full_loss, loss_config=config.losses)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step)

  # Replicate parameters and other state.
  state = flax.jax_utils.replicate(state, devices=jax.local_devices())
  # Initialize gradient accumulation states.
  # We assume accum_grad is a tree matching the gradients; None means no accumulation yet.
  accum_grad = None
  # accum_steps: number of mini-batches accumulated so far.
  accum_steps = 0
  # We need to replicate these for each device.
  accum_grad = flax.jax_utils.replicate(accum_grad)
  accum_steps = flax.jax_utils.replicate(accum_steps)

  del rng  # rng is stored in the state.

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)
  writer.write_hparams(utils.prepare_dict_for_logging(config.to_dict()))

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if jax.process_index() == 0:
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)

  # Create a pmap version of train_step_accum.
  p_train_step_accum = jax.pmap(
      train_step_accum,
      axis_name="batch",
      donate_argnums=(1, 2, 3, 4, 5),
      static_broadcasted_argnums=(0, 6, 7, 8, 9, 10, 11, 12)
  )

  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    if config.num_train_steps == 0:
      with report_progress.timed("eval"):
        evaluate(model, state, eval_ds, loss_fn, eval_metrics_cls, config,
                 writer, step=0)
      with report_progress.timed("checkpoint"):
        ckpt.save(flax.jax_utils.unreplicate(state))
      return

    for step in range(initial_step, config.num_train_steps + 1):
      is_last_step = step == config.num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree_map(np.asarray, next(train_iter))
        (opt, state_vars, rng, metrics_update

