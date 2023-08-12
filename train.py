import timm
import torch
import torchvision
import time
import datetime

from torch import optim
from torch.utils.data import DataLoader
from torch import nn

from timm.utils import accuracy, AverageMeter

from logger import create_logger
from data_loader import get_data_transformer, initialize_mixup
from utils import save_checkpoint
from script import get_training_args


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        criterion,
        scaler,
        epoch,
        lr_scheduler,
        logger,
        device,
        mixup_fn
):
    model.train()

    # train_acc = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    epoch_start = time.time()

    for i, (img_tensor, labels) in enumerate(data_loader):
        batch_start = time.time()
        img_tensor = img_tensor.to(device)
        labels = labels.to(device)
        if mixup_fn:
            img_tensor, labels = mixup_fn(img_tensor, labels)

        # forward
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(img_tensor)
            loss = criterion(out, labels)
            # print(loss)

        # backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()

        # batch_acc = accuracy(out, labels, topk=(1,))
        batch_time.update(time.time() - batch_start, 1)
        loss_meter.update(loss.item(), labels.size(0))
        # train_acc.update(batch_acc[0].item(), labels.size(0))

        current_lr = optimizer.param_groups[0]["lr"]

        if i % (len(data_loader) // 10) == 0:
            logger.info(
                f"Training: Epoch[{epoch + 1}/{100}]\t"
                f"lr:{current_lr:.4f}\t"
                f"batch[{i + 1}/{len(data_loader)}]\t"
                f"batch loss: {loss_meter.val:.6f}({loss_meter.avg:.6f})\t"
                # f"batch_acc: {train_acc.val:.3f} %({train_acc.avg:.3f} %)\t"
            )
            logger.info(f"(data for plot fig)-{loss_meter.val:.6f},{loss_meter.avg:.6f}")

    epoch_time = time.time() - epoch_start
    logger.info(f"Training: Epoch[{epoch + 1}/{100}] training takes {epoch_time:.2f} s")


def val_one_epoch(model, data_loader, epoch, logger, device):
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for i, (img_tensor, labels) in enumerate(data_loader):
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                out = model(img_tensor)

            batch_acc = accuracy(out, labels, topk=(1,))
            acc_meter.update(batch_acc[0].item(), labels.size(0))

        logger.info(f"Testing: Epoch[{epoch + 1}/{100}]\t"
                    f"acc:{acc_meter.avg:.3f} %\n")

    return acc_meter.avg


def initialize_head(model, input_channels, class_nums):
    for param in model.parameters():
        param.requires_grad = False
    model.head = nn.Linear(input_channels, class_nums)
    return model


def main(config):
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(config)
    print(f"Using {device} as device")

    model = timm.create_model(config.model_name, pretrained=True, num_classes=10)
    model = initialize_head(model, 768, 10)
    model = nn.DataParallel(model).to(device)

    current_time = datetime.datetime.now()
    log_name = current_time.strftime('%Y-%m-%d_%H_%M_%S')
    logger = create_logger(config.log_root, log_name)

    # data processing
    training_transformer = get_data_transformer(config, 1)
    training_set = torchvision.datasets.CIFAR10(config.data_root,
                                                train=True, transform=training_transformer, download=True)
    training_loader = DataLoader(training_set,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=True)

    mixup_fn = initialize_mixup(config)

    testing_transformer = get_data_transformer(config, 0)
    testing_set = torchvision.datasets.CIFAR10(config.data_root, train=False,
                                               transform=testing_transformer, download=True)
    testing_loader = DataLoader(testing_set,
                                batch_size=config.batch_size,
                                shuffle=False,
                                drop_last=False)
    # training set
    optimizer = optim.AdamW(model.parameters(), lr=config.opt_lr, weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    # lr changing when [0, 2/5 epochs, 4/5 epochs, epochs]
    # scheduler = lambda t: np.interp([t], [0, config.epochs * 2 // 5, config.epochs * 4 // 5, config.epochs],
    #                                 [0, config.lr_max, config.lr_max / 20.0, 0])[0]
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.max_lr,
                                              steps_per_epoch=len(training_loader), epochs=config.epochs)

    current_best_acc = 0.

    # training loop
    seq_start = time.time()
    for epoch in range(config.epochs):
        train_one_epoch(model, training_loader, optimizer, criterion, scaler, epoch, scheduler,
                        logger, device, mixup_fn)
        current_acc = val_one_epoch(model, testing_loader, epoch, logger, device)

        if current_acc > current_best_acc:
            save_checkpoint(config.checkpoint_root, model, epoch, current_acc)
            logger.info(f"New model has been saved in {config.checkpoint_root}\n")
            current_best_acc = current_acc

    logger.info(f"Training completed! training time:{(time.time() - seq_start):.2f} s")


if __name__ == '__main__':
    # vit_base_patch16_224
    config = get_training_args()
    main(config)
