import os, time

import torch
from torch.utils.data import DataLoader

from utils import mode, setSeed
from utils.checkpoint import load_checkpoint, save_checkpoint
from Hyperparameter.Lip2WavHP import Lip2WavHP
from Logger.Lip2WavLogger import Lip2WavLogger
from Datasets.Lip2WavDataset import Lip2WavDataset, Lip2WavCollate
from Model.Lip2WavTacotron2 import Lip2WavTacotron2
from Loss.Lip2WavLoss import Lip2WavLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # faster due to dynamic input shape
# ====================== step 1/5 前期准备 ====================== #
setSeed(0)
hp = Lip2WavHP()

use_cuda = hp.cuda_available and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# get logger ready
logger = Lip2WavLogger(hp)
logger.log_start_info(device=device)
# ====================== step 2/5 准备数据 ====================== #
# make dataset
collate_fn = Lip2WavCollate(hp, hp.n_frames_per_step)

train_dataset = Lip2WavDataset(hp, "train")
train_loader = DataLoader(train_dataset, num_workers=hp.n_workers, shuffle=True,
						  batch_size=hp.batch_size, pin_memory=hp.pin_mem,
						  drop_last=True, collate_fn=collate_fn)

test_dataset = Lip2WavDataset(hp, "test")
test_loader = DataLoader(test_dataset, num_workers=hp.n_workers, shuffle=False,
						 batch_size=32, pin_memory=hp.pin_mem,
						 drop_last=True, collate_fn=collate_fn)

logger.info("train dataset len = {}".format(len(train_dataset)))
logger.info("test dataset len = {}".format(len(test_dataset)))
# ====================== step 3/5 准备模型 ====================== #
# build model
model = Lip2WavTacotron2(hp)
mode(model, True)
model = model.cuda()
model.train()

loss_function = Lip2WavLoss(hp)

# get optimizer/scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr,
							 betas=hp.betas, eps=hp.eps,
							 weight_decay=hp.weight_decay)
lr_lambda = lambda step: hp.sch_step ** 0.5 * min((step + 1) * hp.sch_step ** -1.5, (step + 1) ** -0.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# load checkpoint
epoch_idx = 1
if hp.load_checkpoint_path != '':
	model, optimizer, epoch_idx, scheduler = load_checkpoint(hp.load_checkpoint_path, model, optimizer, scheduler)
	epoch_idx += 1

# ====================== step 4/5 开始训练 ====================== #
while epoch_idx <= hp.max_iter:
	for batch in train_loader:
		if epoch_idx > hp.max_iter:
			break
		start = time.perf_counter()
		input, target = model.parse_batch(batch)
		predict = model(input)

		# loss
		mel_loss, mel_loss_post, l1_loss, gate_loss = loss_function(predict, target, epoch_idx)

		loss = mel_loss + mel_loss_post + l1_loss + gate_loss
		items = [mel_loss.item(), mel_loss_post.item(), l1_loss.item(), gate_loss.item()]
		# zero grad
		model.zero_grad()

		# backward, grad_norm, and update
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)
		optimizer.step()
		if hp.sch:
			scheduler.step()

		# info
		dur = time.perf_counter() - start
		logger.info('Iter: {} Loss: {:.5f} Grad Norm: {:.5f} {:.1f}s/it'.format(epoch_idx, sum(items), grad_norm, dur))

		# log
		if hp.log_dir != '' and (epoch_idx % hp.iters_per_log == 0):
			learning_rate = optimizer.param_groups[0]['lr']
			logger.log_training_vid(predict, target, items, grad_norm, learning_rate, epoch_idx)

		# sample
		if hp.log_dir != '' and (epoch_idx % hp.iters_per_sample == 0):
			model.eval()
			for i, batch in enumerate(test_loader):
				if i == 0:
					test_input, test_target = model.parse_batch_vid(batch)
					test_predict = model.inference(test_input, 'train')
					logger.sample_training(test_predict, test_target, epoch_idx)
				else:
					break
			model.train()

		# save ckpt
		if hp.checkpoint_dir != '' and (epoch_idx % hp.iters_per_ckpt == 0):
			ckpt_pth = os.path.join(hp.checkpoint_dir, 'ckpt_{}.pt'.format(epoch_idx))
			save_checkpoint(model, optimizer, scheduler, epoch_idx, ckpt_pth)
		epoch_idx += 1

# ====================== step 5/5 保存模型 ====================== #
if hp.log_dir != '':
	logger.close()

logger.log_end_info()