import os
from utils import makedirs
from datetime import datetime
from SubPart import SubParameters
from FaceLandmarks import face_mesh_collections

class HyperParameter():
    hp_name = "Base Hyper Parameters"

    fmc = face_mesh_collections.FMesh_Whole_Face
    num_landmarks = fmc.landmark_count
    save_checkpoint = True
    checkpoint_interval = 1  # 若干 epoch 记录一次checkpoint
    dataloader_shuffle = False
    landmark_dir = "Landmark"
    train_annotate = "Norm_Train"

    # Train
    main_model = "Tacotron2"
    max_iter = 100
    batch_size = 32
    test_batch_size = 24    # 该值越小，验证结果越精确，验证时间越长
    n_frames_per_step = 1  # 每步预测mel帧的数量
    loss_penalty = 1  # mel spec loss penalty

    # log per iteration
    iters_per_log = 50
    iters_per_sample = 500
    iters_per_ckpt = 1000

    # others
    seed = 1
    cropT = 90  # 90
    crop_overlap = 30
    img_size = 96
    fps = 30
    num_init_filters = 24

    landmark_pad = 1000  # 设置为大于 max(video_time)*frame 的数值
    specmask_pad = 1500  # 设置为大于 max(audio_time)*sample_rate/hop_size 的数值

    # scheduled_sampling_method = "none"
    # zero_steps = 5000
    # noise_start_add = 14500
    # noise_end_add = 180000          # 大于 tf_start_decay + tf_decay_steps 即可视为一直添加 noise

    # teacher_force_ratio = 1
    # add_ds_epoch = []
    # main_decayer_params = {
    #     "global_step": 0,
    #     "decay_method_str": "min_cosin",
    #     "init_ratio": 1.0,
    #     "start_decay": 15000,
    #     "decay_steps": 75000,
    #     "alpha": 0,
    # }
    #
    # ns_decay_ratio = 2.0


    ################################
    # Optimizer Parameters         #
    ################################
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-6

    sch = True
    sch_step = 4000

    weight_decay = 1e-6
    grad_clip_thresh = 1.0  # max norm of the gradients
    mask_padding = True

    ################################
    # Server Parameters            #
    ################################
    main_dir = '/Users/jinchenji/Developer/JetBrains/Pycharm/LipLandmark2Wav'
    datasets_dir = '/Users/jinchenji/Developer/Datasets/Lip2Wav/chem'
    result_dir = os.path.join(main_dir, 'Results')

    # CUDA
    use_cuda = False
    pin_mem = True
    n_workers = 8

    ################################
    # CheckPoint Parameters        #
    ################################
    load_checkpoint = False
    load_ckpt_name = "Trainning_Apr27(16-30-53)"
    load_ckpt_epoch = 260
    use_save_lr = False
    load_checkpoint_path = os.path.join(result_dir,
                                        "{}/ckpt_dir/ckpt_{:05d}.pt".format(load_ckpt_name, load_ckpt_epoch))
    ################################
    # Log Parameters               #
    ################################
    save_alignment_image = False
    log_tf_eval = False
    log_train_metric = False
    log_eval_metric = True
    log_tf_metric = False

    ################################
    # Audio Parameters            #
    ################################
    sample_rate = 16000  # 16000

    # Mel Spectrogram
    num_mels = 80       # mel-spec 的数量
    n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
    hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    mel_overlap = 40
    mel_step_size = 241

    # other
    frame_shift_ms = 12.5
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    power = 1.5
    use_lws = False
    ################################
    # Model Parameters             #
    ################################
    # Encoder
    encoder_kernel_size = 5
    encoder_embedding_dim = 384  # encoder_lstm_units
    encoder_n_convolutions = 3   # encoder 中卷积层的数量
    encoder_n_conv_channels = [24, 48, 96, 192]
    p_encoder_dropout = 0.5

    # Decoder
    decoder_rnn_dim = 1024
    decoder_layers = 2
    decoder_lstm_units = 256

    max_decoder_steps = mel_step_size
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # PreNet/PostNet
    prenet_dim = 256
    prenet_layers = [256, 256]

    postnet_embedding_dim = 512
    postnet_n_convolutions = 5
    postnet_kernel_size = 5

    ### Transformer
    n_head = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    transformer_dropout = 0.3
    trans_method = "none"
    layer_norm_eps = 1e-5


    ################################
    # Data Parameters              #
    ################################
    trim_fft_size = 512
    trim_hop_size = 128
    trim_top_db = 23

    signal_normalization = True
    allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
    symmetric_mels = True
    max_abs_value = 4.
    normalize_for_wavenet = True
    clip_for_wavenet = True

    preemphasize = True  # whether to apply filter
    fmin = 55
    fmax = 7600  # To be increased/reduced depending on data.
    griffin_lim_iters = 60

    def __init__(self, annotate=None):
        self.train_start_time = datetime.now().strftime('%b%d(%H-%M-%S)')
        if annotate != None:
            self.train_annotate = annotate
        self.timming_annotate = "{}_{}".format(self.train_annotate, self.train_start_time)

        self.current_result_dir = os.path.join(self.result_dir, self.timming_annotate)
        makedirs(self.current_result_dir)

        self.log_dir = os.path.join(self.current_result_dir, 'logger_dir')
        makedirs(self.log_dir)
        self.checkpoint_dir = os.path.join(self.current_result_dir, 'ckpt_dir')
        makedirs(self.checkpoint_dir)


    def update(self, params: SubParameters):
        for attr in vars(params):
            vars(self)[attr] = vars(params)[attr]