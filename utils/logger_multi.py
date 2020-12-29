import numpy as np
from utils.util import to_arr
from hparams import hparams as hps
from tensorboardX import SummaryWriter
from utils.audio import inv_melspectrogram
from utils.audio_v import inv_mel_spectrogram
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class Tacotron2Logger(SummaryWriter):
	def __init__(self, logdir):
		super(Tacotron2Logger, self).__init__(logdir, flush_secs = 5)

	def log_training(self, reduced_loss, grad_norm, learning_rate, iteration):
			self.add_scalar("training.loss", reduced_loss, iteration)
			self.add_scalar("grad.norm", grad_norm, iteration)
			self.add_scalar("learning.rate", learning_rate, iteration)

	def log_training_vid(self, output, target, reduced_loss, grad_norm, learning_rate, iteration):
			mel_loss, mel_loss_post, l1_loss, gate_loss = reduced_loss
			self.add_scalar("training.mel_loss", mel_loss, iteration)
			self.add_scalar("training.mel_loss_post", mel_loss_post, iteration)
			self.add_scalar("training.l1_loss", l1_loss, iteration)
			self.add_scalar("training.gate_loss", gate_loss, iteration)
			self.add_scalar("grad.norm", grad_norm, iteration)
			self.add_scalar("learning.rate", learning_rate, iteration)

			mel_outputs = to_arr(output[0][0])
			mel_target = to_arr(target[0][0])
			mel_outputs_postnet = to_arr(output[1][0])
			alignments = to_arr(output[3][0]).T

			# plot alignment, mel and postnet output
			self.add_image(
				"alignment",
				plot_alignment_to_numpy(alignments),
				iteration)
			self.add_image(
				"mel_outputs",
				plot_spectrogram_to_numpy(mel_outputs),
				iteration)
			self.add_image(
				"mel_outputs_postnet",
				plot_spectrogram_to_numpy(mel_outputs_postnet),
				iteration)
			self.add_image(
				"mel_target",
				plot_spectrogram_to_numpy(mel_target),
				iteration)

			# save audio
			# try:  # sometimes error
			wav = inv_mel_spectrogram(mel_outputs, hps)
			# wav *= 32767 / max(0.01, np.max(np.abs(wav)))
			# wav /= max(0.01, np.max(np.abs(wav)))
			wav_postnet = inv_mel_spectrogram(mel_outputs_postnet, hps)
			# wav_postnet *= 32767 / max(0.01, np.max(np.abs(wav_postnet)))
			# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
			wav_target = inv_mel_spectrogram(mel_target, hps)
			# wav_target *= 32767 / max(0.01, np.max(np.abs(wav_target)))
			# wav_target /= max(0.01, np.max(np.abs(wav_target)))
			self.add_audio('pred', wav, iteration, hps.sample_rate)
			self.add_audio('pred_postnet', wav_postnet, iteration, hps.sample_rate)
			self.add_audio('target', wav_target, iteration, hps.sample_rate)
			# except:
			# 	pass

	# def sample_training(self, output, target, iteration):
	# 		mel_outputs = to_arr(output[0][0])
	# 		mel_target = to_arr(target[0][0])
	# 		mel_outputs_postnet = to_arr(output[1][0])
	# 		alignments = to_arr(output[2][0]).T
	#
	# 		# plot alignment, mel and postnet output
	# 		self.add_image(
	# 			"alignment_test",
	# 			plot_alignment_to_numpy(alignments),
	# 			iteration)
	# 		self.add_image(
	# 			"mel_outputs_test",
	# 			plot_spectrogram_to_numpy(mel_outputs),
	# 			iteration)
	# 		self.add_image(
	# 			"mel_outputs_postnet_test",
	# 			plot_spectrogram_to_numpy(mel_outputs_postnet),
	# 			iteration)
	# 		self.add_image(
	# 			"mel_target_test",
	# 			plot_spectrogram_to_numpy(mel_target),
	# 			iteration)
	#
	# 		# save audio
	# 		# try: # sometimes error
	# 		wav = inv_mel_spectrogram(mel_outputs, hps)
	# 		wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	# 		# wav /= max(0.01, np.max(np.abs(wav)))
	# 		wav_postnet = inv_mel_spectrogram(mel_outputs_postnet, hps)
	# 		wav_postnet *= 32767 / max(0.01, np.max(np.abs(wav)))
	# 		# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
	# 		wav_target = inv_mel_spectrogram(mel_target, hps)
	# 		wav_target *= 32767 / max(0.01, np.max(np.abs(wav)))
	# 		# wav_target /= max(0.01, np.max(np.abs(wav_target)))
	# 		self.add_audio('pred_test', wav, iteration, hps.sample_rate)
	# 		self.add_audio('pred_postnet_test', wav_postnet, iteration, hps.sample_rate)
	# 		self.add_audio('target_test', wav_target, iteration, hps.sample_rate)
	# 		# except:
	# 		# 	pass

	def sample_training(self, output, target, iteration):
		mel_outputs1 = to_arr(output[0][0])
		mel_target1 = to_arr(target[0][0])
		mel_outputs_postnet1 = to_arr(output[1][0])
		alignments1 = to_arr(output[2][0]).T

		mel_outputs2 = to_arr(output[0][9])
		mel_target2 = to_arr(target[0][9])
		mel_outputs_postnet2 = to_arr(output[1][9])
		alignments2 = to_arr(output[2][9]).T

		# plot alignment, mel and postnet output
		self.add_image(
			"alignment_test1",
			plot_alignment_to_numpy(alignments1),
			iteration)
		self.add_image(
			"mel_outputs_test1",
			plot_spectrogram_to_numpy(mel_outputs1),
			iteration)
		self.add_image(
			"mel_outputs_postnet_test1",
			plot_spectrogram_to_numpy(mel_outputs_postnet1),
			iteration)
		self.add_image(
			"mel_target_test1",
			plot_spectrogram_to_numpy(mel_target1),
			iteration)

		self.add_image(
			"alignment_test2",
			plot_alignment_to_numpy(alignments2),
			iteration)
		self.add_image(
			"mel_outputs_test2",
			plot_spectrogram_to_numpy(mel_outputs2),
			iteration)
		self.add_image(
			"mel_outputs_postnet_test2",
			plot_spectrogram_to_numpy(mel_outputs_postnet2),
			iteration)
		self.add_image(
			"mel_target_test2",
			plot_spectrogram_to_numpy(mel_target2),
			iteration)

		# save audio
		# try: # sometimes error
		wav1 = inv_mel_spectrogram(mel_outputs1, hps)
		# wav1 *= 32767 / max(0.01, np.max(np.abs(wav1)))
		# wav /= max(0.01, np.max(np.abs(wav)))
		wav_postnet1 = inv_mel_spectrogram(mel_outputs_postnet1, hps)
		# wav_postnet1 *= 32767 / max(0.01, np.max(np.abs(wav_postnet1)))
		# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
		wav_target1 = inv_mel_spectrogram(mel_target1, hps)
		# wav_target1 *= 32767 / max(0.01, np.max(np.abs(wav_target1)))
		# wav_target /= max(0.01, np.max(np.abs(wav_target)))
		self.add_audio('pred_test1', wav1, iteration, hps.sample_rate)
		self.add_audio('pred_postnet_test1', wav_postnet1, iteration, hps.sample_rate)
		self.add_audio('target_test1', wav_target1, iteration, hps.sample_rate)
		# except:
		# 	pass

		# save audio
		# try: # sometimes error
		wav2 = inv_mel_spectrogram(mel_outputs2, hps)
		# wav2 *= 32767 / max(0.01, np.max(np.abs(wav2)))
		# wav /= max(0.01, np.max(np.abs(wav)))
		wav_postnet2 = inv_mel_spectrogram(mel_outputs_postnet2, hps)
		# wav_postnet2 *= 32767 / max(0.01, np.max(np.abs(wav_postnet2)))
		# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
		wav_target2 = inv_mel_spectrogram(mel_target2, hps)
		# wav_target2 *= 32767 / max(0.01, np.max(np.abs(wav_target2)))
		# wav_target /= max(0.01, np.max(np.abs(wav_target)))
		self.add_audio('pred_test2', wav2, iteration, hps.sample_rate)
		self.add_audio('pred_postnet_test2', wav_postnet2, iteration, hps.sample_rate)
		self.add_audio('target_test2', wav_target2, iteration, hps.sample_rate)
		# except:
		# 	pass
