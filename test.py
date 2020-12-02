# from inference import load_model, infer_vid
import torch
from shutil import copy
from torchvision import transforms
from PIL import Image
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm
import sys, cv2, os, pickle, argparse, subprocess

from model.model import Tacotron2
from hparams import hparams as hps
import utils.audio_v as audio
from utils.util import mode, to_var, to_arr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Generator(object):
	def __init__(self, model):
		super(Generator, self).__init__()

		self.synthesizer = model.eval()


	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			height, width, channels = img.shape
			path = Path(fname)
			fparent = str(path.parent.parent.parent)
			############ transform
			if fparent[-3:] == 'new':
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = Image.fromarray(img)
				crop = transforms.CenterCrop((height // 2, width // 2))
				img = crop(img)
				img = np.array(img)
				# Convert RGB to BGR
				img = img[:, :, ::-1].copy()
			if img is None:
				raise FileNotFoundError('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (hps.img_size, hps.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		return images

	def vc(self, sample, outfile):
		# hp = sif.hparams
		images = sample['images']
		all_windows = []
		i = 0
		while i + hps.T <= len(images):
			all_windows.append(images[i : i + hps.T])
			i += hps.T - hps.overlap
		all_windows.append(images[i: len(images)])

		for window_idx, window_fnames in enumerate(all_windows):
			images = self.read_window(window_fnames)
			# s = self.synthesizer.synthesize_spectrograms(images)[0] ######
			mel_outputs, mel_outputs_postnet, alignments = infer_vid(images, self.synthesizer, mode='test')

			s = mel_outputs_postnet.squeeze(0).contiguous().cpu().detach().numpy()
			if window_idx == 0:
				mel = s
			else:
				mel = np.concatenate((mel, s[:, hps.mel_overlap:]), axis=1)
		wav = audio.inv_mel_spectrogram(mel, hps)

		audio.save_wav(wav, outfile, sr=hps.sample_rate)


def get_image_list(split, data_root):
	filelist = []
	with open(os.path.join(data_root, '{}.txt'.format(split))) as vidlist:
		for vid_id in vidlist:
			vid_id = vid_id.strip()
			filelist.extend(list(glob(os.path.join(data_root, 'preprocessed', vid_id, '*/*.jpg'))))
			filelist.extend(list(glob(os.path.join(data_root, 'preprocessed_new', vid_id, '*/*.jpg'))))
	return filelist


def get_testlist(data_root):
	test_images = get_image_list('test', data_root)
	# print(data_root)
	# print(test_images)
	print('{} hours is available for testing'.format(len(test_images) / (hps.fps * 3600.)))
	test_vids = {}
	for x in test_images:
		x = x[:x.rfind('/')]
		test_vids[x] = True
	return list(test_vids.keys())

def to_sec(idx):
	frame_id = idx + 1
	sec = frame_id / float(hps.fps)
	return sec

def contiguous_window_generator(vidpath):
	frames = glob(os.path.join(vidpath, '*.jpg'))
	if len(frames) < hps.T: return

	ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
	sortedids = sorted(ids)
	end_idx = 0
	start = sortedids[end_idx]

	while end_idx < len(sortedids):
		while end_idx < len(sortedids):
			if end_idx == len(sortedids) - 1:
				if sortedids[end_idx] + 1 - start >= hps.T:
					yield ((to_sec(start), to_sec(sortedids[end_idx])), 
					[os.path.join(vidpath, '{}.jpg'.format(x)) for x in range(start, sortedids[end_idx] + 1)])
				return
			else:
				if sortedids[end_idx] + 1 == sortedids[end_idx + 1]:
					end_idx += 1
				else:
					if sortedids[end_idx] + 1 - start >= hps.T:
						yield ((to_sec(start), to_sec(sortedids[end_idx])), 
						[os.path.join(vidpath, '{}.jpg'.format(x)) for x in range(start, sortedids[end_idx] + 1)])
					break
		
		end_idx += 1
		start = sortedids[end_idx]


def load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth)
	model = Tacotron2()
	model.load_state_dict(ckpt_dict['model'])
	for name, param in model.named_parameters():
		print(name)
	model = mode(model, True).eval()
	return model


def infer_vid(inputs, model, mode='train'):
	mel_outputs, mel_outputs_postnet, _, alignments = model.inference(inputs, mode)
	return (mel_outputs, mel_outputs_postnet, alignments)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--data_dir", help="Speaker folder path", required=True)
	parser.add_argument('-r', "--results_dir", help="Speaker folder path", required=True)
	parser.add_argument('--checkpoint', help="Path to trained checkpoint", required=True)
	parser.add_argument("--preset", help="Speaker-specific hyper-params", type=str, required=False)
	args = parser.parse_args()

	#todo add speaker-specific parameters
	# with open(args.preset) as f:
	# 	sif.hparams.parse_json(f.read())

	# sif.hparams.set_hparam('eval_ckpt', args.checkpoint)
	
	videos = get_testlist(args.data_root)

	if not os.path.isdir(args.results_root):
		os.mkdir(args.results_root)

	GTS_ROOT = os.path.join(args.results_root, 'gts/')
	WAVS_ROOT = os.path.join(args.results_root, 'wavs/')
	files_to_delete = []
	if not os.path.isdir(GTS_ROOT):
		os.mkdir(GTS_ROOT)
	else:
		files_to_delete = list(glob(GTS_ROOT + '*'))
	if not os.path.isdir(WAVS_ROOT):
		os.mkdir(WAVS_ROOT)
	else:
		files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
	for f in files_to_delete: os.remove(f)


	tacotron = load_model(args.checkpoint)
	model = Generator(tacotron)

	template = 'ffmpeg -y -loglevel panic -ss {} -i {} -to {} -strict -2 {}'
	for vid in tqdm(videos):
		vidpath = vid + '/'
		for (ss, es), images in tqdm(contiguous_window_generator(vidpath)):

			sample = {}
			sample['images'] = images

			vidname = vid.split('/')[-2] + '_' + vid.split('/')[-1]
			outfile = '{}{}_{}:{}.wav'.format(WAVS_ROOT, vidname, ss, es)
			try:
				model.vc(sample, outfile)
			except KeyboardInterrupt:
				exit(0)
			except Exception as e:
				print(e)
				continue

			command = template.format(ss, vidpath + 'audio.wav', es, 
									'{}{}_{}:{}.wav'.format(GTS_ROOT, vidname, ss, es))

			subprocess.call(command, shell=True)
