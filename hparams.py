from text import symbols


class hparams:
	################################
	# Data Parameters              #
	################################
	text_cleaners=['english_cleaners']

	################################
	# Audio                        #
	################################
	num_mels = 80
	num_freq = 1025
	sample_rate = 16000
	frame_length_ms = 50
	frame_shift_ms = 12.5
	preemphasis = 0.97
	min_level_db = -100
	ref_level_db = 20
	power = 1.5
	gl_iters = 100

	################################
	# Model Parameters             #
	################################
	n_symbols = len(symbols)
	symbols_embedding_dim = 512

	# Encoder parameters
	encoder_kernel_size = 5

	# Decoder parameters
	n_frames_per_step = 2
	decoder_rnn_dim = 1024
	prenet_dim = 256
	max_decoder_steps = 120
	gate_threshold = 0.5
	p_attention_dropout = 0.1
	p_decoder_dropout = 0.1

	# Attention parameters
	attention_rnn_dim = 1024
	attention_dim = 128

	# Location Layer parameters
	attention_location_n_filters = 32
	attention_location_kernel_size = 31

	# Mel-post processing network parameters
	postnet_embedding_dim = 512
	postnet_kernel_size = 5
	postnet_n_convolutions = 5

	################################
	# Train                        #
	################################
	is_cuda = True
	pin_mem = True
	n_workers = 8
	lr = 2e-3
	betas = (0.9, 0.999)
	eps = 1e-6
	sch = True
	sch_step = 4000
	max_iter = 1e6
	batch_size = 40
	iters_per_log = 50
	iters_per_sample = 500
	iters_per_ckpt = 1000
	weight_decay = 1e-6
	grad_clip_thresh = 1.0
	mask_padding = False #### 바꿨음
	p = 10 # mel spec loss penalty
	eg_text = 'Make America great again!'

	############# added
	iscrop = True
	encoder_embedding_dim = 384  # encoder_lstm_units
	encoder_n_convolutions = 5  # enc_conv_num_blocks

	num_init_filters= 24

	prenet_layers= [256, 256]
	decoder_layers= 2
	decoder_lstm_units= 256

	tacotron_teacher_forcing_start_decay= 29000
	tacotron_teacher_forcing_decay_steps= 130000

	T= 90 #90
	overlap= 15
	mel_overlap= 40
	mel_step_size= 240
	img_size = 96
	fps= 30


	use_lws = False
	# Mel spectrogram
	n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
	hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)

	# M-AILABS (and other datasets) trim params (these parameters are usually correct for any
	# data, but definitely must be tuned for specific speakers)
	trim_fft_size = 512
	trim_hop_size = 128
	trim_top_db = 23

	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization = True # Only relevant if mel_normalization = True
	symmetric_mels = True
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
	# faster and cleaner convergence)
	max_abs_value = 4.
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
	# be too big to avoid gradient explosion,
	# not too small for fast convergence)
	normalize_for_wavenet = True
	# whether to rescale to [0, 1] for wavenet. (better audio quality)
	clip_for_wavenet = True
	# whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)

	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
	# levels. Also allows for better G&L phase reconstruction)
	preemphasize = True # whether to apply filter

	fmin = 55
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax = 7600  # To be increased/reduced depending on data.

	# Griffin Lim
	# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters = 60
# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
###########################################################################################################################################