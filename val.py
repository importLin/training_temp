import os

import numpy as np
import timm
import torch

from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as T
from logger import create_logger


def val_one_epoch(model, data_loader, fig_logger, device, epoch):
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for i, (img_tensor, labels) in enumerate(data_loader):
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)

            out = model(img_tensor)
            batch_acc = accuracy(out, labels, topk=(1,))
            acc_meter.update(batch_acc[0].item(), labels.size(0))

            # if i % (len(data_loader) // 4) == 0:
            #     fig_logger.info(f"testing acc-{acc_meter.avg:.3f}")

        fig_logger.info(f"Epoch:{epoch} testing acc-{acc_meter.avg:.3f}")

        return acc_meter.avg


def dict_remapping(ori_dict, obj_dict):
    ori_keys = [k for k, v in ori_dict.items()]
    obj_val = [v for k, v in obj_dict.items()]
    for i in range(len(ori_keys)):
        key = ori_keys[i]
        ori_dict[key] = obj_val[i]

    return ori_dict


def main():
    print("start")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    command = "gaussian_noise_test"
    logger = create_logger(f"{command} tesing", "log")

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
    # weights = torch.load("checkpoints/ViT_imgn1k/best/best_model_95.56.pth")["weights"]
    model.load_state_dict(torch.load("/home/linhaiwei/PycharmProjects/encryption_sim/weights/ori_weights.pth"))
    # remap_dict = dict_remapping(model.state_dict(), weights)
    # model.load_state_dict(remap_dict)
    model = model.to(device)

    cifar10_mean = (0.5000, 0.5000, 0.5000)
    cifar10_std = (0.5000, 0.5000, 0.5000)

    sigma = 0.3
    epochs = 10
    transformer = T.Compose([
        T.ToTensor(),
        AddGaussianNoise(sigma),
        # AddShotNoise(sigma=sigma),

        # AddSpNoise(sigma)
        # AddUniformNoise(sigma),
        # AddSpNoise(sigma),
        T.Normalize(cifar10_mean, cifar10_std),

    ])
    # logger.info(f"gaussian noise added--sigma{sigma}")
    logger.info(f"{command} testing(norm first)-- sigma{sigma}")

    image_root = "data/cifar_plaintext_224"
    plain_set = Testingset(image_root, transformer)
    # testing_set = torchvision.datasets.CIFAR10("data/trainning_img", train=False, transform=transformer, download=False)
    plain_loader = DataLoader(
        plain_set,
        batch_size=512,
        shuffle=True,
        drop_last=True
    )

    acc_list1 = []
    for epoch in range(epochs):
        acc = val_one_epoch(model, plain_loader, logger, device, epoch)
        acc_list1.append(acc)

    logger.info(f"plaintext used--avg acc-{np.mean(acc_list1):.3f}\n")

    model.load_state_dict(torch.load("/home/linhaiwei/PycharmProjects/encryption_sim/weights/e_vit.pth"))
    model = model.to(device)
    cipher_root = "data/cifar_ciphertext_224"
    cipher_set = Testingset(cipher_root, transformer)
    cipher_loader = DataLoader(
        cipher_set,
        batch_size=512,
        shuffle=True,
        drop_last=True
    )
    acc_list2 = []
    for epoch in range(epochs):
        acc = val_one_epoch(model, cipher_loader, logger, device, epoch)
        acc_list2.append(acc)

    logger.info(f"ciphertext used--avg acc-{np.mean(acc_list2):.3f}")


if __name__ == '__main__':
    main()



