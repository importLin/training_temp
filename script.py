import argparse


def get_training_args():
    parser = argparse.ArgumentParser('Training and evaluation script')
    parser.add_argument("--model-name", type=str, default="convmixer_1024_20_ks9_p14")

    # root
    parser.add_argument("--data-root", type=str, default="imgs")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoint")
    parser.add_argument("--log-root", type=str, default="log")

    # hyperparameter
    parser.add_argument("--opt_lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument('--clip-norm', action='store_true')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)

    # data augmentation
    parser.add_argument("--mixup_used", action='store_true', default=True)
    # parser.add_argument('--scale', default=0.75, type=float)
    parser.add_argument('--reprob', default=0.25, type=float)
    parser.add_argument('--ra-m', default=8, type=int)
    parser.add_argument('--ra-n', default=1, type=int)
    parser.add_argument('--jitter', default=0.1, type=float)

    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.5,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument("--num_classes", type=int, default=10)

    return parser.parse_args()


def main(args):
    print(args.model_name)


if __name__ == '__main__':
    config = get_training_args()
    main(config)
