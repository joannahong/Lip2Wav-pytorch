import numpy as np
from utils import to_arr
from Logger import TacotronLogger
from Hyperparameter.Lip2WavHP import Lip2WavHP as hp
from tensorboardX import SummaryWriter
from utils.audio import inv_mel_spectrogram
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class Lip2WavLogger(TacotronLogger):
	def __init__(self, hp):
		super(Lip2WavLogger, self).__init__(hp)

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
		wav = inv_mel_spectrogram(mel_outputs, hp)
		wav *= 32767 / max(0.01, np.max(np.abs(wav)))
		# wav /= max(0.01, np.max(np.abs(wav)))
		wav_postnet = inv_mel_spectrogram(mel_outputs_postnet, hp)
		wav_postnet *= 32767 / max(0.01, np.max(np.abs(wav_postnet)))
		# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
		wav_target = inv_mel_spectrogram(mel_target, hp)
		wav_target *= 32767 / max(0.01, np.max(np.abs(wav_target)))
		# wav_target /= max(0.01, np.max(np.abs(wav_target)))
		self.add_audio('pred', wav, iteration, hp.sample_rate)
		self.add_audio('pred_postnet', wav_postnet, iteration, hp.sample_rate)
		self.add_audio('target', wav_target, iteration, hp.sample_rate)

	# except:
	# 	pass

	def sample_training(self, output, target, iteration):
		mel_target = to_arr(target[0][0])

		mel_outputs = to_arr(output[0][0])
		mel_outputs_postnet = to_arr(output[1][0])
		alignments = to_arr(output[3][0]).T

		# plot alignment, mel and postnet output
		self.add_image(
			"alignment_test",
			plot_alignment_to_numpy(alignments),
			iteration)
		self.add_image(
			"mel_outputs_test",
			plot_spectrogram_to_numpy(mel_outputs),
			iteration)
		self.add_image(
			"mel_outputs_postnet_test",
			plot_spectrogram_to_numpy(mel_outputs_postnet),
			iteration)
		self.add_image(
			"mel_target_test",
			plot_spectrogram_to_numpy(mel_target),
			iteration)

		# save audio
		# try: # sometimes error
		wav = inv_mel_spectrogram(mel_outputs, hp)
		# 			wav *= 32767 / max(0.01, np.max(np.abs(wav)))
		# wav /= max(0.01, np.max(np.abs(wav)))
		wav_postnet = inv_mel_spectrogram(mel_outputs_postnet, hp)
		# 			wav_postnet *= 32767 / max(0.01, np.max(np.abs(wav_postnet)))
		# wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
		wav_target = inv_mel_spectrogram(mel_target, hp)
		# 			wav_target *= 32767 / max(0.01, np.max(np.abs(wav_target)))
		# wav_target /= max(0.01, np.max(np.abs(wav_target)))
		self.add_audio('pred_test', wav, iteration, hp.sample_rate)
		self.add_audio('pred_postnet_test', wav_postnet, iteration, hp.sample_rate)
		self.add_audio('target_test', wav_target, iteration, hp.sample_rate)
# except:
# 	pass

