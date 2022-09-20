import torch


def load_checkpoint(ckpt_pth, model, optimizer, scheduler, map_loaction=None):

    ckpt_dict = torch.load(ckpt_pth, map_location=map_loaction)

    model.load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    scheduler.load_state_dict(ckpt_dict['scheduler'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, scheduler, iteration



def save_checkpoint(model, optimizer, scheduler, iteration, ckpt_pth):
    ckpt_content = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iteration': iteration,
    }
    torch.save(ckpt_content, ckpt_pth)
