from PIL import Image
from torchvision import transforms as T
import os
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from timm.data.mixup import Mixup


class Cifar(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.img_name = os.listdir(data_root)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img_path = os.path.join(self.data_root, img_name)

        img_data = Image.open(img_path)
        img_tensor = self.transform(img_data)
        label = int(img_name.split("_")[0])
        return img_tensor, label

    def __len__(self):
        return len(self.img_name)


def get_data_transformer(args, training_mode):
    if training_mode:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)

        transformer = T.Compose([
            # transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
            T.ColorJitter(args.jitter, args.jitter, args.jitter),
            T.ToTensor(),
            T.Normalize(cifar10_mean, cifar10_std),
            T.RandomErasing(p=args.reprob)
        ])

    else:
        cifar10_mean = (0.5000, 0.5000, 0.5000,)
        cifar10_std = (0.5000, 0.5000, 0.5000,)
        transformer = T.Compose([
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(cifar10_mean, cifar10_std)
        ])

    return transformer


def initialize_mixup(args):
    mixup_args = dict(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.num_classes)

    return Mixup(**mixup_args)


def main():
    initialize_mixup()


if __name__ == '__main__':
    main()
