import os
from Hyperparameter import HyperParameter

class Lip2WavHP(HyperParameter):
	def __init__(self):
		super(Lip2WavHP, self).__init__()
		self.main_dir = '/Users/jinchenji/Developer/JetBrains/Pycharm/LipLandmark2Wav'
		self.datasets_dir = '/Users/jinchenji/Developer/Datasets/Lip2Wav/chem'

		self.load_checkpoint = False
		self.load_ckpt_name = "Trainning_Apr27(16-30-53)"
		self.load_ckpt_epoch = 260
		self.load_checkpoint_path = os.path.join(self.checkpoint_dir,
												 "{}/ckpt_{:05d}.pt".format(self.load_ckpt_name, self.load_ckpt_epoch))
		self.use_save_lr = False
		self.n_frames_per_step = 2

		self.num_init_filters= 24

		# self.tacotron_teacher_forcing_start_decay= 29000
		# self.tacotron_teacher_forcing_decay_steps= 130000

		self.cropT = 90 #90
		self.img_size = 96
		self.fps = 30

		################################
		# Audio Parameters            #
		################################
		self.num_mels = 80
		self.sample_rate = 16000
		self.frame_shift_ms = 12.5
		self.preemphasis = 0.97
		self.min_level_db = -100
		self.ref_level_db = 20
		self.power = 1.5
		self.use_lws = False
		# Mel spectrogram
		self.n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
		self.hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
		self.win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
		self.mel_overlap= 40
		self.mel_step_size= 240

		################################
		# Model Parameters             #
		################################

		# Encoder parameters
		self.encoder_kernel_size = 5
		self.encoder_embedding_dim = 384  # encoder_lstm_units
		self.encoder_n_convolutions = 5  # enc_conv_num_blocks

		# Decoder parameters
		self.decoder_rnn_dim = 1024
		self.prenet_dim = 256
		self.max_decoder_steps = 120
		self.gate_threshold = 0.5
		self.p_attention_dropout = 0.1
		self.p_decoder_dropout = 0.1
		self.decoder_layers= 2
		self.decoder_lstm_units= 256

		# Attention parameters
		self.attention_rnn_dim = 1024
		self.attention_dim = 128

		# Location Layer parameters
		self.attention_location_n_filters = 32
		self.attention_location_kernel_size = 31

		# PreNet/PostNet
		self.prenet_layers= [256, 256]

		self.postnet_embedding_dim = 512
		self.postnet_kernel_size = 5
		self.postnet_n_convolutions = 5

		# Train                        #
		self.use_cuda = True
		self.pin_mem = True
		self.n_workers = 8
		self.lr = 2e-3
		self.betas = (0.9, 0.999)
		self.eps = 1e-6
		self.sch = True
		self.sch_step = 4000
		self.max_iter = 1e6
		self.batch_size = 40
		self.iters_per_log = 50
		self.iters_per_sample = 500
		self.iters_per_ckpt = 1000
		self.weight_decay = 1e-6
		self.grad_clip_thresh = 1.0
		self.mask_padding = True
		self.loss_penalty = 10 # mel spec loss penalty

		############# added




		# M-AILABS (and other datasets) trim params (these parameters are usually correct for any
		# data, but definitely must be tuned for specific speakers)
		self.trim_fft_size = 512
		self.trim_hop_size = 128
		self.trim_top_db = 23

		# Mel and Linear spectrograms normalization/scaling and clipping
		self.signal_normalization = True
		# Whether to normalize mel spectrograms to some predefined range (following below parameters)
		self.allow_clipping_in_normalization = True # Only relevant if mel_normalization = True
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
		self.preemphasize = True # whether to apply filter

		self.fmin = 55
		# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
		# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
		self.fmax = 7600  # To be increased/reduced depending on data.

		# Griffin Lim
		# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
		self.griffin_lim_iters = 60
# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
###########################################################################################################################################
