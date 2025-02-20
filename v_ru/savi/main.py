from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from ml_collections import config_flags

from savi.lib import trainer

import tensorflow as tf

# set flags 
FLAGS = flags.FLAGS

# pick config 
config_flags.DEFINE_config_file("config", None, "Config file.")
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("jax_backend_target", None, "JAX backend target to use.")
flags.mark_flags_as_required(["config", "workdir"])


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax.config.update("jax_xla_backend", "tpu_driver")
    jax.config.update("jax_backend_target", FLAGS.jax_backend_target)

  logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")


  trainer.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  app.run(main)
