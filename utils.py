import torch
import os


def save_checkpoint_best(config, model, optimizer, epoch, max_accuracy, logger):
    logger.info(f"new max acc: {max_accuracy:.2f}")
    weights = model.state_dict()
    save_state = {'weights': weights,
                  'optimizer': optimizer.state_dict()
                  }

    model_root = config.checkpoint_root + "/best"

    file_name = f"best_model_{max_accuracy:.2f}.pth"
    save_path = os.path.join(model_root, file_name)
    logger.info(f"checkpoint saving...")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved!!!\n")


def save_checkpoint(checkpoint_root, model, epoch, current_accuracy):

    file_name = f"epoch{epoch+1}_{current_accuracy:.2f}.pth"
    save_path = os.path.join(checkpoint_root, file_name)
    torch.save(model.state_dict(), save_path)

