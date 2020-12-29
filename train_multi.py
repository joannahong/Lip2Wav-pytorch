import os
import time
import torch
import argparse
import numpy as np
from utils.util import mode
from hparams import hparams as hps
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from utils.logger_multi import Tacotron2Logger
from utils.dataset import VideoMelLoader, VMcollate
from utils.util import to_var, get_mask_from_lengths, zero_grad

from model.model import Encoder3D, Decoder, Postnet, Tacotron2, Tacotron2Loss
from model.memory_network import Memory

from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

### memory 없는 버전

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def parse_batch(batch):
	text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
	text_padded = to_var(text_padded).long()
	input_lengths = to_var(input_lengths).long()
	max_len = torch.max(input_lengths.data).item()
	mel_padded = to_var(mel_padded).float()
	gate_padded = to_var(gate_padded).float()
	output_lengths = to_var(output_lengths).long()

	return (
		(text_padded, input_lengths, mel_padded, max_len, output_lengths),
		(mel_padded, gate_padded))


def parse_batch_vid(batch):
	vid_padded, input_lengths, mel_padded, gate_padded, target_lengths, split_infos, embed_targets, idx = batch
	vid_padded = to_var(vid_padded).float()
	input_lengths = to_var(input_lengths).float()
	mel_padded = to_var(mel_padded).float()
	gate_padded = to_var(gate_padded).float()
	target_lengths = to_var(target_lengths).float()
	idx = to_var(idx).float()

	max_len_vid = split_infos[0].data.item()
	max_len_target = split_infos[1].data.item()

	mel_padded = to_var(mel_padded).float()

	return (
		(vid_padded, input_lengths, mel_padded, max_len_vid, target_lengths, embed_targets, idx),
		(mel_padded, gate_padded))


def parse_output(outputs, output_lengths=None):
	if hps.mask_padding and output_lengths is not None:
		mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
		mask = mask.expand(hps.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
		mask = mask.permute(1, 0, 2)  # (B, 80, T)

		outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
		outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
		slice = torch.arange(0, mask.size(2), hps.n_frames_per_step)
		outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)

	return outputs


def prepare_dataloaders_vid(fdir):
	trainset = VideoMelLoader(fdir, hps, "train")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return train_loader


def prepare_dataloaders_vid_test(fdir):
	trainset = VideoMelLoader(fdir, hps, "test")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	test_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = False,
							  batch_size = 40, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return test_loader


def load_checkpoint(ckpt_pth, models, optimizer):
	encoder, decoder, postnet = models
	ckpt_dict = torch.load(ckpt_pth)
	encoder.load_state_dict(ckpt_dict['encoder'])
	decoder.load_state_dict(ckpt_dict['decoder'])
	postnet.load_state_dict(ckpt_dict['postnet'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return encoder, decoder, postnet, optimizer, iteration


def save_checkpoint(models, optimizer, iteration, ckpt_pth):
	encoder, decoder, postnet = models
	torch.save({'encoder': encoder.state_dict(),
				'decoder': decoder.state_dict(),
				'postnet': postnet.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def train(args):
	# build model
	#todo
	encoder = Encoder3D(hps)
	decoder = Decoder()
	postnet = Postnet()
	# model = Tacotron2()

	if args.dataparallel:
		encoder = DataParallel(encoder)
		decoder = DataParallel(decoder)
		postnet = DataParallel(postnet)

	encoder = mode(encoder, True)
	decoder = mode(decoder, True)
	postnet = mode(postnet, True)
	models = [encoder, decoder, postnet]

	optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()},
								  {'params': postnet.parameters()}],
								 lr = hps.lr, betas = hps.betas,
								 eps = hps.eps, weight_decay = hps.weight_decay)

	criterion = Tacotron2Loss()
	
	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		encoder, decoder, postnet, optimizer, iteration = load_checkpoint(args.ckpt_pth, models, optimizer)
		iteration += 1 # next iteration is iteration+1
	
	# get scheduler
	if hps.sch:
		lr_lambda = lambda step: hps.sch_step**0.5*min((step+1)*hps.sch_step**-1.5, (step+1)**-0.5)
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


	# make dataset
	# train_loader = prepare_dataloaders(args.data_dir)
	train_loader = prepare_dataloaders_vid(args.data_dir)
	# train_loader = prepare_dataloaders_vid_test(args.data_dir)
	test_loader = prepare_dataloaders_vid_test(args.data_dir)

	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir+ current_time):
			os.makedirs(args.log_dir+ current_time)
			os.chmod(args.log_dir+ current_time, 0o775)
		logger = Tacotron2Logger(args.log_dir+ current_time)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir+ current_time):
		os.makedirs(args.ckpt_dir+ current_time)
		os.chmod(args.ckpt_dir+ current_time, 0o775)
	
	encoder.train()
	decoder.train()
	postnet.train()
	# ================ MAIN TRAINNIG LOOP! ===================
	while iteration <= hps.max_iter:
		for batch in train_loader:
			if iteration > hps.max_iter:
				break
			start = time.perf_counter()
			# x, y = model.parse_batch(batch)
			x, y = parse_batch_vid(batch)

			vid_inputs, vid_lengths, mels, max_len, output_lengths, refs, idx = x
			mels = mels.cuda()
			refs = refs.cuda()
			vid_lengths, output_lengths = vid_lengths.data, output_lengths.data
			embedded_inputs = vid_inputs.type(torch.FloatTensor)
			encoder_outputs = encoder(embedded_inputs.cuda(), vid_lengths.cuda())  # [bs x 25 x encoder_embedding_dim]

			refs_repeat = refs.transpose(1,2).contiguous()  # [bs x 256 x 1] --> [bs x 1 x 256]
			refs_repeat = refs_repeat.repeat(1, hps.T, 1)  # [bs x 1 x 256] --> [bs x 25 x 256]
			encoder_outputs = torch.cat((encoder_outputs, refs_repeat), dim=2)
			mel_outputs, gate_outputs, alignments = decoder(encoder_outputs, mels, memory_lengths=vid_lengths)  # decoder

			mel_outputs_postnet = postnet(mel_outputs)  # postnet
			mel_outputs_postnet = mel_outputs + mel_outputs_postnet
			y_pred = parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)
			# y_pred = model(x)
			########

			# loss
			mel_loss, mel_loss_post, l1_loss, gate_loss = criterion(y_pred, y, iteration)

			loss = mel_loss + mel_loss_post + l1_loss + gate_loss
			items = [mel_loss.item(), mel_loss_post.item(), l1_loss.item(), gate_loss.item()]
			# zero grad
			encoder.zero_grad()
			decoder.zero_grad()
			postnet.zero_grad()

			# backward, grad_norm, and update
			loss.backward()
			grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), hps.grad_clip_thresh)
			grad_norm_dec = torch.nn.utils.clip_grad_norm_(decoder.parameters(), hps.grad_clip_thresh)
			grad_norm_post = torch.nn.utils.clip_grad_norm_(postnet.parameters(), hps.grad_clip_thresh)
			grad_norm = grad_norm_enc + grad_norm_dec + grad_norm_post

			optimizer.step()

			if hps.sch:
				scheduler.step()

			# info
			dur = time.perf_counter()-start
			print('Iter: {} Loss: {:.5f} Grad Norm: {:.5f} {:.1f}s/it'.format(
				iteration, sum(items), grad_norm, dur))
			
			# # log
			# if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
			# 	learning_rate = optimizer.param_groups[0]['lr']
			# 	logger.log_training(item, grad_norm, learning_rate, iteration)

			# log vid
			if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training_vid(y_pred, y, items, grad_norm, learning_rate, iteration)

			# sample
			#todo!!!!!!!!!!!!!!!!!!!!!
			if args.log_dir != '' and (iteration % hps.iters_per_sample == 0):
			# if True:
				encoder.eval()
				decoder.eval()
				postnet.eval()
				for i, batch in enumerate(test_loader):
					if i == 0:
						x_test, y_test = parse_batch_vid(batch)
						vid_inputs, vid_lengths, mels, max_len, output_lengths, refs, idx = x_test
						refs = refs.cuda()
						embedded_inputs = vid_inputs.type(torch.FloatTensor)
						if args.dataparallel:
							encoder_outputs = encoder.module.inference(embedded_inputs.cuda())
						else:
							encoder_outputs = encoder.inference(embedded_inputs.cuda())

						refs_repeat = refs.transpose(1, 2).contiguous()  # [bs x 256 x 1] --> [bs x 1 x 256]
						refs_repeat = refs_repeat.repeat(1, hps.T, 1)  # [bs x 1 x 256] --> [bs x 25 x 256]
						encoder_outputs = torch.cat((encoder_outputs, refs_repeat), dim=2)
						if args.dataparallel:
							mel_outputs, gate_outputs, alignments = decoder.module.inference(encoder_outputs)
						else:
							mel_outputs, gate_outputs, alignments = decoder.inference(encoder_outputs)
						mel_outputs_postnet = postnet(mel_outputs)

						mel_outputs_postnet = mel_outputs + mel_outputs_postnet
						outputs = parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
						mel_outputs, mel_outputs_postnet, _, alignments = outputs
						output = (mel_outputs, mel_outputs_postnet, alignments)

						# output = infer_vid(x_test, model)
						logger.sample_training(output, y_test, iteration)
					else:
						break
				encoder.train()
				decoder.train()
				postnet.train()

			# save ckpt
			if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
				models = [encoder, decoder, postnet]
				ckpt_pth = os.path.join(args.ckpt_dir+ current_time, 'ckpt_{}'.format(iteration))
				save_checkpoint(models, optimizer, iteration, ckpt_pth)

			iteration += 1
	if args.log_dir != '':
		logger.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# path
	parser.add_argument('-d', '--data_dir', type = str, default = 'data',
						help = 'directory to load data')
	parser.add_argument('-l', '--log_dir', type = str, default = 'log/',
						help = 'directory to save tensorboard logs')
	parser.add_argument('-cd', '--ckpt_dir', type = str, default = 'ckpt/',
						help = 'directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'path to load checkpoints')
	parser.add_argument("--dataparallel", default=True)

	args = parser.parse_args()


	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False # faster due to dynamic input shape

	train(args)