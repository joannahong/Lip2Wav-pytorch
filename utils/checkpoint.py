import torch


def load_checkpoint(ckpt_pth, model, optimizer, scheduler, map_loaction=None):

    ckpt_dict = torch.load(ckpt_pth, map_location=map_loaction)

    model.load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    scheduler.load_state_dict(ckpt_dict['scheduler'])
    epoch = ckpt_dict['epoch']
    return model, optimizer, scheduler, epoch



def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_pth):
    ckpt_content = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckpt_content, ckpt_pth)
