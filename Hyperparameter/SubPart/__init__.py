import os
class SubParameters():
    def __init__(self):
        pass


class ServerParameters(SubParameters):
    def __init__(self):
        super(ServerParameters, self).__init__()
        # Server Parameters
        self.main_dir = '/Users/Developer/LipLandmark2Wav'
        self.datasets_dir = '/Users/Developer/Datasets/Lip2Wav/chem'
        self.log_dir = os.path.join(self.main_dir, 'Logger/logger_file')
        self.checkpoint_dir = os.path.join(self.main_dir, 'Checkpoint/ckpt_file')


class ModelParameters(SubParameters):
    def __init__(self, mel_step_size):
        super(ModelParameters, self).__init__()
        # Encoder
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3  # encoder 中卷积层的数量
        self.encoder_n_conv_channels = [24, 48, 96, 192]
        self.encoder_embedding_dim = 384  # encoder_lstm_units

        # Decoder
        self.n_frames_per_step = 1  # 每步预测mel帧的数量
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = mel_step_size

        self.gate_threshold = 0.5
        self.p_encoder_dropout = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.decoder_layers = 2
        self.decoder_lstm_units = 256

        ### Attention
        self.attention_rnn_dim = 1024
        self.attention_dim = 128
        ###### Location Layer
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        ### Transformer
        self.n_head = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.transformer_dropout = 0.3
        self.trans_method = "none"
        self.layer_norm_eps = 1e-5

        # PostNet
        self.prenet_layers = [256, 256]

        self.postnet_n_convolutions = 5
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5

        self.n_symbols = 148
        self.symbols_embedding_dim = 512


class AudioParameters(SubParameters):
    def __init__(self):
        super(AudioParameters, self).__init__()
        self.sample_rate = 16000  # 16000
        self.num_mels = 80  # mel-spec 的数量
        self.frame_shift_ms = 12.5
        self.preemphasis = 0.97
        self.min_level_db = -100
        self.ref_level_db = 20
        self.power = 1.5

        # Mel spectrogram
        self.n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
        self.hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        self.win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)

        self.mel_overlap = 40
        self.mel_step_size = 241
        # padding size
        self.landmark_pad = 1000  # 设置为大于 max(video_time)*frame 的数值
        self.specmask_pad = 1500  # 设置为大于 max(audio_time)*sample_rate/hop_size 的数值

        self.num_freq = 1025
        self.frame_length_ms = 50
        # gl_iters = 100


class OptimizerParameters(SubParameters):
    def __init__(self):
        super(OptimizerParameters, self).__init__()
        # optimizer Parameters
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-6
        self.weight_decay = 1e-6
        self.momentum = 0.5

        # scheduler Parameters
        self.sch = True
        self.sch_step = 4000


class CheckPointParameters(SubParameters):
    def __init__(self, checkpoint_dir):
        super(CheckPointParameters, self).__init__()
        self.save_checkpoint = True
        self.checkpoint_interval = 1  # 若干 epoch 记录一次checkpoint

        self.load_checkpoint = False
        self.load_ckpt_name = "Trainning_Apr27(16-30-53)"
        self.load_ckpt_epoch = 260
        self.load_checkpoint_path = os.path.join(checkpoint_dir,
                                                 "{}/ckpt_{:05d}.pt".format(self.load_ckpt_name, self.load_ckpt_epoch))
        self.use_save_lr = False


class LogParameters(SubParameters):
    def __init__(self):
        super(LogParameters, self).__init__()
        self.log_interval = 1
        self.save_alignment_image = False
        self.log_tf_eval = False

        self.log_train_metric = False
        self.log_eval_metric = True
        self.log_tf_metric = False


class DataParameters(SubParameters):
    def __init__(self):
        super(DataParameters, self).__init__()
        ################################
        # Data Parameters              #
        ################################
        self.img_size = 96
        self.text_cleaners = ['english_cleaners']

        self.use_lws = False

        # M-AILABS (and other datasets) trim params (these parameters are usually correct for any
        # data, but definitely must be tuned for specific speakers)
        self.trim_fft_size = 512
        self.trim_hop_size = 128
        self.trim_top_db = 23

        # Mel and Linear spectrograms normalization/scaling and clipping
        self.signal_normalization = True
        # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        self.allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
        self.symmetric_mels = True
        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
        # faster and cleaner convergence)
        self.max_abs_value = 4.
        # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
        # be too big to avoid gradient explosion,
        # not too small for fast convergence)
        self.normalize_for_wavenet = True
        # whether to rescale to [0, 1] for wavenet. (better audio quality)
        self.clip_for_wavenet = True
        # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)

        # Contribution by @begeekmyfriend
        # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
        # levels. Also allows for better G&L phase reconstruction)
        self.preemphasize = True  # whether to apply filter

        # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
        # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
        self.fmin = 55
        self.fmax = 7600  # To be increased/reduced depending on data.

        # Griffin Lim
        # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
        self.griffin_lim_iters = 60

        # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
        ###########################################################################################################################################
