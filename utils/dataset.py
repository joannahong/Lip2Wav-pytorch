import os
import torch
import random
import numpy as np
from hparams import hparams as hps
from torch.utils.data import Dataset
from utils.audio import load_wav, melspectrogram
from torchvision import transforms
from PIL import Image

import cv2
from glob import glob
from pathlib import Path
from os.path import dirname, join, basename, isfile
random.seed(0)


def files_to_list(fdir):
	f_list = []
	with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
			f_list.append([wav_path, parts[1]])
	return f_list

def get_image_list(split, data_root):
	filelist = []
	with open(os.path.join(data_root, '{}.txt'.format(split))) as vidlist:
		for vid_id in vidlist:
			vid_id = vid_id.strip()
			filelist.extend(list(glob(os.path.join(data_root, 'preprocessed', vid_id, '*/*.jpg'))))
			filelist.extend(list(glob(os.path.join(data_root, 'preprocessed_new', vid_id, '*/*.jpg'))))
	return filelist

class VideoMelLoader(torch.utils.data.Dataset):
	def __init__(self, fdir, hparams, split='train'):
		self._hparams = hparams
		# self.filelist = {'train': self._hparams.all_images, 'test': self._hparams.all_test_images}
		# if split == 'train':
		# 	self.filelist = self._hparams.all_images
		# else:
		# 	self.filelist = self._hparams.all_test_images
		self.filelist = get_image_list(split, fdir)

		self.test_steps = 2

		# pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		# explicitely setting the padding to a value that doesn"t originally exist in the spectogram
		# to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.
		# Mark finished sequences with 1s
		self._token_pad = 1.

	def get_window(self, center_frame):
		center_id = self.get_frame_id(center_frame)
		vidname = dirname(center_frame)
		if self._hparams.T % 2:
			window_ids = range(center_id - self._hparams.T // 2, center_id + self._hparams.T // 2 + 1)
		else:
			window_ids = range(center_id - self._hparams.T // 2, center_id + self._hparams.T // 2)

		window_fnames = list()
		for frame_id in window_ids:
			frame = join(vidname, '{}.jpg'.format(frame_id))

			if not isfile(frame):
				return None
			window_fnames.append(frame)
		return window_fnames

	def crop_audio_window(self, spec, center_frame):
		# estimate total number of frames from spec (num_features, T)
		# num_frames = (T x hop_size * fps) / sample_rate
		start_frame_id = self.get_frame_id(center_frame) - self._hparams.T // 2
		total_num_frames = int((spec.shape[0] * self._hparams.hop_size * self._hparams.fps) / self._hparams.sample_rate)

		start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
		end_idx = start_idx + self._hparams.mel_step_size
		return spec[start_idx: end_idx, :]

	def get_frame_id(self, frame):
		return int(basename(frame).split('.')[0])

	def __getitem__(self, index):
		img_name = self.filelist[index]
		window_fnames = self.get_window(img_name)

		mel = np.load(os.path.join(os.path.dirname(img_name), 'mels.npz'))['spec'].T
		mel = self.crop_audio_window(mel, img_name)

		window = []
		if type(window_fnames) == list:
			for fname in window_fnames:
				img = cv2.imread(fname)
				height, width, channels = img.shape
				path = Path(fname)
				fparent = str(path.parent.parent.parent)
				############ transform
				if fparent[-3:] == 'new':
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					img = Image.fromarray(img)
					crop = transforms.CenterCrop((height//2, width//2))
					img = crop(img)
					img = np.array(img)
					# Convert RGB to BGR
					img = img[:, :, ::-1].copy()
				try:
					img = cv2.resize(img, (self._hparams.img_size, self._hparams.img_size))
				except:
					continue

				window.append(img)
			x = np.asarray(window) / 255.
		else:
			x = None
		embed_target = np.zeros([256], dtype=np.float32)
		return x, mel.T, embed_target, len(mel)

	def __len__(self):
		return len(self.filelist)

class VMcollate():
	def __init__(self, hparams, n_frames_per_step):
		self.n_frames_per_step = n_frames_per_step
		self._hparams = hparams

	def __call__(self, batch):
		# Right zero-pad all one-hot text sequences to max input length
		vid_refined, mel_refined, target_lengths, embed_refined = [], [], [], []

		for i, (vid, aud, emb_tar, aud_len) in enumerate(batch):
			if not vid is None:
				vid = torch.Tensor(vid)
				aud = torch.Tensor(aud)
				vid_refined.append(vid)
				mel_refined.append(aud)
				embed_refined.append(emb_tar)
				target_lengths.append(aud_len)

		input_lengths, ids_sorted_decreasing = torch.sort(
			torch.LongTensor([len(v) for v in vid_refined]),
			dim=0, descending=True)

		bsz = len(input_lengths)
		max_input_len = input_lengths[0]
		# vid_padded = torch.LongTensor(len(batch), max_input_len) #todo
		vid_padded = torch.Tensor(bsz, 3, max_input_len, self._hparams.img_size, self._hparams.img_size) #todo
		vid_padded.zero_()

		for i in range(len(ids_sorted_decreasing)):
			vid_sorted = vid_refined[ids_sorted_decreasing[i]]
			vid_sorted = vid_sorted.permute(3, 0, 1, 2)
			vid_padded[i, :, :vid_sorted.size(1), :, :] = vid_sorted

		# Right zero-pad mel-spec
		# num_mels = vid_refined[1].size(0) #todo check
		num_mels = self._hparams.num_mels
		max_target_len = max([m.size(1) for m in mel_refined])

		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0

		# include mel padded and gate padded
		mel_padded = torch.FloatTensor(bsz, num_mels, max_target_len)
		mel_padded.zero_()
		gate_padded = torch.FloatTensor(bsz, max_target_len)
		gate_padded.zero_()
		target_lengths = torch.LongTensor(bsz)
		for i in range(len(ids_sorted_decreasing)):
			mel = mel_refined[ids_sorted_decreasing[i]]
			mel_padded[i, :, :mel.size(1)] = mel
			gate_padded[i, mel.size(1)-1:] = 1
			target_lengths[i] = mel.size(1)

		embed_targets = torch.LongTensor(embed_refined)
		split_infos = torch.LongTensor([max_input_len, max_target_len])

		return vid_padded, input_lengths, mel_padded, gate_padded, target_lengths, split_infos, embed_targets

# class VMcollate():
# 	def __init__(self, hparams, n_frames_per_step):
# 		self.n_frames_per_step = n_frames_per_step
# 		self._hparams = hparams
#
# 	def __call__(self, batch):
# 		# Right zero-pad all one-hot text sequences to max input length
#
# 		input_lengths, ids_sorted_decreasing = torch.sort(
# 			torch.LongTensor([len(x[0]) for x in batch]),
# 			dim=0, descending=True)
#
# 		max_input_len = input_lengths[0]
#
# 		# vid_padded = torch.LongTensor(len(batch), max_input_len) #todo
# 		vid_padded = torch.LongTensor(len(batch), 3, max_input_len, self._hparams.img_size, self._hparams.img_size) #todo
# 		vid_padded.zero_()
#
# 		for i in range(len(ids_sorted_decreasing)):
# 			vid = batch[ids_sorted_decreasing[i]][0]
# 			vid_padded[i, 3, :vid.size(0), self._hparams.img_size, self._hparams.img_size] = vid
#
# 		# Right zero-pad mel-spec
# 		num_mels = batch[0][1].size(0)
# 		max_target_len = max([x[1].size(1) for x in batch])
# 		if max_target_len % self.n_frames_per_step != 0:
# 			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
# 			assert max_target_len % self.n_frames_per_step == 0
#
# 		# include mel padded and gate padded
# 		mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
# 		mel_padded.zero_()
# 		gate_padded = torch.FloatTensor(len(batch), max_target_len)
# 		gate_padded.zero_()
# 		target_lengths = torch.LongTensor(len(batch))
# 		for i in range(len(ids_sorted_decreasing)):
# 			mel = batch[ids_sorted_decreasing[i]][1]
# 			mel_padded[i, :, :mel.size(1)] = mel
# 			gate_padded[i, mel.size(1)-1:] = 1
# 			target_lengths[i] = mel.size(1)
#
# 		embed_targets = torch.LongTensor([x[2] for x in batch])
# 		split_infos = torch.LongTensor([max_input_len, max_target_len])
#
# 		# return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
# 		return vid_padded, input_lengths, mel_padded, gate_padded, target_lengths, split_infos, embed_targets
#
# class ljdataset(Dataset):
# 	def __init__(self, fdir):
# 		self.f_list = files_to_list(fdir)
# 		random.shuffle(self.f_list)
#
# 	def get_mel_text_pair(self, filename_and_text):
# 		filename, text = filename_and_text[0], filename_and_text[1]
# 		text = self.get_text(text)
# 		mel = self.get_mel(filename)
# 		return (text, mel)
#
# 	def get_mel(self, filename):
# 		wav = load_wav(filename)
# 		mel = melspectrogram(wav).astype(np.float32)
# 		return torch.Tensor(mel)
#
# 	def get_text(self, text):
# 		text_norm = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
# 		return text_norm
#
# 	def __getitem__(self, index):
# 		return self.get_mel_text_pair(self.f_list[index])
#
# 	def __len__(self):
# 		return len(self.f_list)
#
#
# class ljcollate():
# 	def __init__(self, n_frames_per_step):
# 		self.n_frames_per_step = n_frames_per_step
#
# 	def __call__(self, batch):
# 		# Right zero-pad all one-hot text sequences to max input length
# 		input_lengths, ids_sorted_decreasing = torch.sort(
# 			torch.LongTensor([len(x[0]) for x in batch]),
# 			dim=0, descending=True)
# 		max_input_len = input_lengths[0]
#
# 		text_padded = torch.LongTensor(len(batch), max_input_len)
# 		text_padded.zero_()
# 		for i in range(len(ids_sorted_decreasing)):
# 			text = batch[ids_sorted_decreasing[i]][0]
# 			text_padded[i, :text.size(0)] = text
#
# 		# Right zero-pad mel-spec
# 		num_mels = batch[0][1].size(0)
# 		max_target_len = max([x[1].size(1) for x in batch])
# 		if max_target_len % self.n_frames_per_step != 0:
# 			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
# 			assert max_target_len % self.n_frames_per_step == 0
#
# 		# include mel padded and gate padded
# 		mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
# 		mel_padded.zero_()
# 		gate_padded = torch.FloatTensor(len(batch), max_target_len)
# 		gate_padded.zero_()
# 		output_lengths = torch.LongTensor(len(batch))
# 		for i in range(len(ids_sorted_decreasing)):
# 			mel = batch[ids_sorted_decreasing[i]][1]
# 			mel_padded[i, :, :mel.size(1)] = mel
# 			gate_padded[i, mel.size(1)-1:] = 1
# 			output_lengths[i] = mel.size(1)
#
# 		return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
