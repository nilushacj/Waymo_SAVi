import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.seed_data = True

    config.batch_size = 64

    #Changed to BRNO
    config.data = ml_collections.ConfigDict({
        "name":"waymo_video/video6:1.0.0",
        "data_dir":"/scratch/eng/t212-amlab/waymo/waymo-video-tfds",
        "shuffle_buffer_size": config.batch_size * 8 #??
    })
    
    config.camera_select = 'camera_FRONT'
    config.max_instances = 10
    config.bbox_thresh = 0.005

    config.num_slots = config.max_instances + 1  # Only used for metrics. # originally missed
    config.logging_min_n_colors = config.max_instances # originally missed


    config.preproc_train = [
        f"from_waymo_open(camera_key='{config.camera_select}', max_num_bboxes={config.max_instances}, bbox_threshold={config.bbox_thresh})",
        "random_resized_crop(height=128, width=192, min_object_covered=0.75)",
        #"lidar_to_depth()",  # NOTE: For quantizing, we want to ignore 0s.
        "transform_depth(transform='log_plus')", #"transform_depth(transform='log_plus')",
    ]

    config.preproc_eval = [
        f"from_waymo_open(camera_key='{config.camera_select}', max_num_bboxes={config.max_instances}, bbox_threshold={config.bbox_thresh})",
        "resize_small(128)",
        "crop_or_pad(height=128, width=192)",
        #"lidar_to_depth()",
        "transform_depth(transform='log_plus')"
    ]

    config.eval_slice_size = 6
    #config.eval_slice_keys = [
    #    "video", "segmentations", "flow", "boxes", "depth"
    #]

    config.eval_slice_keys = [
        "video", "boxes", "depth"
    ]

    #config.targets = {"flow": 3, "depth": 1}
    config.targets = {"depth": 1}
    config.losses = ml_collections.ConfigDict({
        f"recon_{target}": {"loss_type": "recon", "key": target}
        for target in config.targets})


    config.model = ml_collections.ConfigDict({
        "module": "savi.modules.SAVi",

        # Encoder.
        "encoder": ml_collections.ConfigDict({
            "module": "savi.modules.FrameEncoder",
            "reduction": "spatial_flatten",

            "backbone": ml_collections.ConfigDict({
                "module": "savi.modules.ResNet34",
                "num_classes": None,
                "axis_name": "time",
                "norm_type": "group",
                "small_inputs": True
            }),
            "pos_emb": ml_collections.ConfigDict({
                "module": "savi.modules.PositionEmbedding",
                "embedding_type": "linear",
                "update_type": "project_add",
                "output_transform": ml_collections.ConfigDict({
                    "module": "savi.modules.MLP",
                    "hidden_size": 64,
                    "layernorm": "pre"
                }),
            }),
            # Transformer.
            "output_transform": ml_collections.ConfigDict({
                "module": "savi.modules.Transformer",
                "num_layers": 4,
                "num_heads": 4,
                "qkv_size": 16 * 4,
                "mlp_size": 1024,
                "pre_norm": True,
            }),
        }),

        # Corrector.
        "corrector": ml_collections.ConfigDict({
            "module": "savi.modules.SlotAttention",
            "num_iterations": 1,
            "qkv_size": 256,
        }),

        # Predictor.
        "predictor": ml_collections.ConfigDict({
            "module": "savi.modules.TransformerBlock",
            "num_heads": 4,
            "qkv_size": 256,
            "mlp_size": 1024
        }),

        # Initializer.
        "initializer": ml_collections.ConfigDict({
            "module": "savi.modules.CoordinateEncoderStateInit",
            "prepend_background": True,
            "center_of_mass": False,
            "embedding_transform": ml_collections.ConfigDict({
                "module": "savi.modules.MLP",
                "hidden_size": 256,
                "output_size": 128,
                "layernorm": None
            }),
        }),

        # Decoder.
        "decoder": ml_collections.ConfigDict({
            "module":
                "savi.modules.SpatialBroadcastDecoder",
            "resolution": (8, 12),  # Update if data resol. or strides change. #DONE
            "early_fusion": True,
            "backbone": ml_collections.ConfigDict({
                "module": "savi.modules.CNN",
                "features": [64, 64, 64, 64],
                "kernel_size": [(5, 5), (5, 5), (5, 5), (5, 5)],
                "strides": [(2, 2), (2, 2), (2, 2), (2, 2)],
                "layer_transpose": [True, True, True, True]
            }),
            "pos_emb": ml_collections.ConfigDict({
                "module": "savi.modules.PositionEmbedding",
                "embedding_type": "linear",
                "update_type": "project_add"
            }),
            "target_readout": ml_collections.ConfigDict({
                "module": "savi.modules.Readout",
                "keys": list(config.targets),
                "readout_modules": [ml_collections.ConfigDict({
                    "module": "savi.modules.MLP",
                    "num_hidden_layers": 0,
                    "hidden_size": 0, "output_size": config.targets[k]})
                                    for k in config.targets],
            }),
        }),
        "decode_corrected": True,
        "decode_predicted": False,  # Disable prediction decoder to save memory.
    })

    # Define which video-shaped variables to log/visualize.
    config.debug_var_video_paths = {
        "recon_masks": "SpatialBroadcastDecoder_0/alphas",
    }
    for k in config.targets:
        config.debug_var_video_paths.update({
            f"{k}_recon": f"SpatialBroadcastDecoder_0/{k}_combined"})

    # Define which attention matrices to log/visualize.
    config.debug_var_attn_paths = {
        "corrector_attn": "SlotAttention_0/InvertedDotProductAttention_0/GeneralizedDotProductAttention_0/attn"
    }

    # Widths of attention matrices (for reshaping to image grid).
    config.debug_var_attn_widths = {
        "corrector_attn": 16,
    }
    
    config.learning_rate = 2e-4
    config.warmup_steps = 2500
    config.num_train_steps = 500000

    config.max_grad_norm = 0.05 
    config.log_loss_every_steps = 50 
    config.eval_every_steps = 1000 
    config.checkpoint_every_steps = 1000 #5000 

    config.train_metrics_spec = {
      "loss": "loss",
      "ari": "ari",
      "ari_nobg": "ari_nobg",
    }
    config.eval_metrics_spec = {
        "eval_loss": "loss",
        "eval_ari": "ari",
        "eval_ari_nobg": "ari_nobg",
    }

    config.conditioning_key = "boxes"

    return config
