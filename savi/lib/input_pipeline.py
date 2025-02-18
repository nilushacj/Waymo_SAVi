from typing import Tuple, List, Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
from clu import deterministic_data

import ml_collections
import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = Array
from savi.lib import preprocessing
from clu import preprocess_spec
import sys
import cv2
import numpy as np
def preprocess_example(features:Dict[str, tf.Tensor],
                       preprocess_strs:List[str])->Dict[str, tf.Tensor]:
  """Processes a single data example.

  Args:
    features: A dictionary containing the tensors of a single data example.
    preprocess_strs: List of strings, describing one preprocessing operation
      each, in clu.preprocess_spec format.

  Returns:
    Dictionary containing the preprocessed tensors of a single data example.
  """
  # 1. This gets all the classes defined in the preprocessing script. 
  # 2. The returned class names are in underscore format, e.g. VideoFromTfds -> video_from_tfds
  # 3. All_ops is an array of paired elements ([('add_temporal_axis', <class 'savi.lib.preprocessing.AddTemporalAxis'>), ('central_crop', <class 'savi.lib.preprocessing.CentralCrop'>),....]) 
  all_ops = preprocessing.all_ops()
  #print(f' --------- ALL_OPS ---------: ************* {all_ops} ************')
  
  # Parses all ops from above (see SS for printed strings)
  preprocess_fn = preprocess_spec.parse("|".join(preprocess_strs), all_ops)
  #print(f' --------- PARSED ALL_OPS ---------: ************* {preprocess_fn} ************')

  return preprocess_fn(features)  # pytype: disable=bad-return-type  # allow-recursive-types


def get_batch_dims(global_batch_size: int) -> List[int]:
  """Gets the first two axis sizes for data batches.

  Args:
    global_batch_size: Integer, the global batch size (across all devices).

  Returns:
    List of batch dimensions

  Raises:
    ValueError if the requested dimensions don't make sense with the
      number of devices.
  """
  num_local_devices = jax.local_device_count()
  if global_batch_size % jax.host_count() != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisble with {jax.host_count()}.")
  per_host_batch_size = global_batch_size // jax.host_count()
  if per_host_batch_size % num_local_devices != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisible with {jax.host_count()} hosts with a per host "
                     f"batch size of {per_host_batch_size} and "
                     f"{num_local_devices} local devices. ")
  return [num_local_devices, per_host_batch_size // num_local_devices]
  

def create_datasets(config: ml_collections.ConfigDict,
                    data_rng: PRNGKey) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  
  # ---- For BRNO ----
  dataset_builder = tfds.builder(config.data.name, data_dir=config.data.data_dir)
  batch_dims = get_batch_dims(config.batch_size)

  #print(f" ----------- INFO: {dataset_builder.info} -----------")

  #---- Set training ----
  train_split = tfds.split_for_jax_process('train', drop_remainder=True)
  #print(f" ----------- train_split:{train_split} ------------")

  #eval_split  = tfds.split_for_jax_process('test', drop_remainder=False)
  #print(f" ----------- eval_split:{eval_split} ------------")

  batch_dims = get_batch_dims(config.batch_size)

  # -- Preprocessing --
  train_preprocess_fn = functools.partial(
    preprocess_example, preprocess_strs=config.preproc_train
  )
  #print(f" ----------- train_preprocess_fn:{train_preprocess_fn} ------------")

  # -- Set train split name --
  train_split = tfds.split_for_jax_process('train', drop_remainder=True)

  # -- Function to create standard input pipeline (preprocess, shuffle, batch)
  train_ds = deterministic_data.create_dataset(
    dataset_builder=dataset_builder, #as_dataset() method is needed
    split=train_split, # ++ OK
    batch_dims=batch_dims, # ++ OK
    rng=data_rng, #a JAX random PRNG used for shuffling, ++OK
    preprocess_fn=train_preprocess_fn, #function to preprocess all individual samples (python dictionary of tensors)
    cache=False, #whether to cache the unprocessed dataset to memory, ++OK
    num_epochs=None, #repeat dataset forever (config has num_train_steps parameter used in trainer), ++ OK
    shuffle=True, # ++ OK
    shuffle_buffer_size=config.data.shuffle_buffer_size#number of examples in the shuffle buffer # ++ OK
  )


  # # TODO:check thiss
  # for batch in train_ds.take(1):
  #   # Suppose each element of the batch is a dictionary.
  #   # And assume 'video' has shape (batch_size, num_frames, H, W, 3)
  #   print(f"TOOK ONE DATA SAMPLE")
  #   video_batch = batch["video"][0][0]
  #   dets_batch = batch["boxes"][0][0]

  #   cam_video = np.array(video_batch.numpy(), dtype=np.uint8)

  #   print(video_batch.shape)
  #   print(dets_batch.shape)


  #   # DRAW RGB CAM bounding boxes
  #   converted_bboxes_rgb = dets_batch.numpy()
  #   for i in range(converted_bboxes_rgb.shape[0]):
  #     for j in range(converted_bboxes_rgb.shape[1]):
  #       temp_detection = converted_bboxes_rgb[i][j]
  #       assert len(temp_detection)==4, f"Wrong length of detection: {temp_detection}"
  #       if temp_detection[-1]!=0:
  #         x1 = int(temp_detection[1]*cam_video[i].shape[1])
  #         y1 = int(temp_detection[0]*cam_video[i].shape[0])
  #         x2 = int(temp_detection[3]*cam_video[i].shape[1])
  #         y2 = int(temp_detection[2]*cam_video[i].shape[0])
  #         cv2.rectangle(cam_video[i], (x1, y1), (x2, y2), (0,255,0), 2)

  #   fp_1_imgs = np.hstack(cam_video)
  #   cv2.imwrite(f'./test_images/test_fp.png', fp_1_imgs)
      
  #   # For example, extract the first sample in the batch.
  #   # You might also extract a specific frame: for instance, the first frame.
  #   #sample_video = video_batch[0]        # shape: (num_frames, H, W, 3)
  #   #first_frame = sample_video[0]          # shape: (H, W, 3)
    
  #   # Convert tensor to numpy array.
  #   #frame_np = first_frame.numpy()
  #   break
    
  # sys.exit()

  # ---- Set validation ----

  #-- Set preprocessing function --
  eval_preprocess_fn = functools.partial(
    preprocess_example, preprocess_strs=config.preproc_eval
  )

  # -- Set validation split --
  eval_split = tfds.split_for_jax_process('validation', drop_remainder=True)

  # -- Function to create standard input pipeline (preprocess, shuffle, batch)
  eval_ds = deterministic_data.create_dataset(
    dataset_builder=dataset_builder, #as_dataset() method is needed
    split=eval_split,
    batch_dims=batch_dims,
    rng=None, #a JAX random PRNG used for shuffling (None means no shuffling)
    preprocess_fn=eval_preprocess_fn, #function to preprocess all individual samples (python dictionary of tensors)
    cache=False, #whether to cache the unprocessed dataset to memory
    num_epochs=1, #repeat dataset forever (config has num_train_steps parameter used in trainer)
    shuffle=False,
    pad_up_to_batches="auto" #process entire dataset. Auto means to derive dataset from batch dims and cardinality such that pad_up_to_batches * batch_dims = cardinality 
  )


  return train_ds, eval_ds

