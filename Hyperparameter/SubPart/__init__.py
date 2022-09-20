import os
class SubParameters():
    def __init__(self):
        pass


class ServerParameters(SubParameters):
    def __init__(self):
        super(ServerParameters, self).__init__()
        # Server Parameters
        self.main_dir = '/Users/jinchenji/Developer/JetBrains/Pycharm/LipLandmark2Wav'
        self.datasets_dir = '/Users/jinchenji/Developer/Datasets/Lip2Wav/chem'
        self.result_dir = os.path.join(self.main_dir, 'Results')

        # CUDA
        self.use_cuda = False
        self.pin_mem = True
        self.n_workers = 8

class ModelParameters(SubParameters):
    def __init__(self, mel_step_size):
        super(ModelParameters, self).__init__()
        # Encoder
        self.encoder_kernel_size = 5
        self.encoder_embedding_dim = 384  # encoder_lstm_units
        self.encoder_n_convolutions = 3  # encoder 中卷积层的数量
        self.encoder_n_conv_channels = [24, 48, 96, 192]
        self.p_encoder_dropout = 0.5

        # Decoder
        self.decoder_rnn_dim = 1024
        self.decoder_layers = 2
        self.decoder_lstm_units = 256

        self.max_decoder_steps = mel_step_size
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # PreNet/PostNet
        self.prenet_dim = 256
        self.prenet_layers = [256, 256]

        self.postnet_embedding_dim = 512
        self.postnet_n_convolutions = 5
        self.postnet_kernel_size = 5

        ### Transformer
        self.n_head = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.transformer_dropout = 0.3
        self.trans_method = "none"
        self.layer_norm_eps = 1e-5

class AudioParameters(SubParameters):
    def __init__(self):
        super(AudioParameters, self).__init__()
        self.sample_rate = 16000  # 16000

        # Mel Spectrogram
        self.num_mels = 80  # mel-spec 的数量
        self.n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
        self.hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        self.win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        self.mel_overlap = 40
        self.mel_step_size = 241

        # other
        self.frame_shift_ms = 12.5
        self.preemphasis = 0.97
        self.min_level_db = -100
        self.ref_level_db = 20
        self.power = 1.5
        self.use_lws = False


class OptimizerParameters(SubParameters):
    def __init__(self):
        super(OptimizerParameters, self).__init__()
        # optimizer Parameters
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-6

        self.sch = True
        self.sch_step = 4000

        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0  # max norm of the gradients
        self.mask_padding = True


class CheckPointParameters(SubParameters):
    def __init__(self, result_dir):
        super(CheckPointParameters, self).__init__()
        self.load_checkpoint = False
        self.load_ckpt_name = "Trainning_Apr27(16-30-53)"
        self.load_ckpt_epoch = 260
        self.use_save_lr = False
        self.load_checkpoint_path = os.path.join(result_dir,
                                                 "{}/ckpt_dir/ckpt_{:05d}.pt".format(self.load_ckpt_name, self.load_ckpt_epoch))


class LogParameters(SubParameters):
    def __init__(self):
        super(LogParameters, self).__init__()
        self.save_alignment_image = False
        self.log_tf_eval = False
        self.log_train_metric = False
        self.log_eval_metric = True
        self.log_tf_metric = False


class DataParameters(SubParameters):
    def __init__(self):
        super(DataParameters, self).__init__()
        self.trim_fft_size = 512
        self.trim_hop_size = 128
        self.trim_top_db = 23

        self.signal_normalization = True
        self.allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
        self.symmetric_mels = True
        self. max_abs_value = 4.
        self.normalize_for_wavenet = True
        self.clip_for_wavenet = True

        self.preemphasize = True  # whether to apply filter
        self.fmin = 55
        self.fmax = 7600  # To be increased/reduced depending on data.
        self.griffin_lim_iters = 60#
