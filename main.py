import os, time
import torch
from torch.utils.data import DataLoader

from utils import setSeed, mode

from arguments import arguments as arg
from utils.checkpoint import load_checkpoint, save_checkpoint

# ====================== step 1/5 前期准备 ====================== #
hp = arg.main_hp
setSeed(hp.seed)

# get logger ready
logger = arg.main_logger(hp=hp)

use_cuda = hp.cuda_available and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

logger.log_start_info(device=device)
# ====================== step 2/5 准备数据 ====================== #
train_dataset = arg.dataset(hp=hp, splits="train")
test_dataset = arg.dataset(hp=hp, splits="test")

collate_fn = arg.collate_fn(hp=hp)
train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hp.test_batch_size, collate_fn=collate_fn, shuffle=False)

logger.info("train dataset len = {}".format(len(train_dataset)))
logger.info("test dataset len = {}".format(len(test_dataset)))
# ====================== step 3/5 准备模型 ====================== #
model = arg.main_model(hp=hp, device=device, logger=logger)
model = mode(model, hp, model=True)

loss_function = arg.loss_fn(hp, logger)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay)
lr_lambda = lambda step: hp.sch_step ** 0.5 * min((step + 1) * hp.sch_step ** -1.5, (step + 1) ** -0.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# load Checkpoint
start_epoch = 1
if hp.load_checkpoint:
    model, optimizer, scheduler, start_epoch = load_checkpoint(hp.load_checkpoint_path,
                                                               model, optimizer, scheduler,
                                                               map_loaction=device)
    start_epoch += 1

# ====================== step 4/5 开始训练 ====================== #
iter_idx = start_epoch
while iter_idx <= hp.max_iter:
    for batch in train_loader:
        if iter_idx > hp.max_iter:
            break
        start = time.perf_counter()
        input, target = model.parse_batch(batch)
        predict = model(input)

        # loss
        mel_loss, mel_loss_post, l1_loss, gate_loss = loss_function(predict, target, iter_idx)

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
        logger.info('Iter: {} Loss: {:.5f} Grad Norm: {:.5f} {:.1f}s/it'.format(iter_idx, sum(items), grad_norm, dur))

        # log
        if hp.log_dir != '' and (iter_idx % hp.iters_per_log == 0):
            learning_rate = optimizer.param_groups[0]['lr']
            logger.log_training_vid(predict, target, items, grad_norm, learning_rate, iter_idx)

        # sample
        if hp.log_dir != '' and (iter_idx % hp.iters_per_sample == 0):
            model.eval()
            for i, batch in enumerate(test_loader):
                if i == 0:
                    test_input, test_target = model.parse_batch_vid(batch)
                    test_predict = model.inference(test_input, 'train')
                    logger.sample_training(test_predict, test_target, iter_idx)
                else:
                    break
            model.train()

        # save ckpt
        if hp.checkpoint_dir != '' and (iter_idx % hp.iters_per_ckpt == 0):
            ckpt_pth = os.path.join(hp.checkpoint_dir, 'ckpt_{}.pt'.format(iter_idx))
            save_checkpoint(model, optimizer, scheduler, iter_idx, ckpt_pth)
        iter_idx += 1

# ====================== step 5/5 保存模型 ====================== #
logger.log_end_info()