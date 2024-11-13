import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, default=0.6)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--name', type=str, default='iE2348C_1286')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='NHP')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--remove', type=str, default="80%")
    return parser.parse_args()