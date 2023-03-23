import argparse
import torch
import clip
from xmodel import XModel
from pytorch_pretrained_gans import make_gan


def train(args):
    """
    Train a model according to specifications in args.
    """
    # Claim the device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("No GPU found, switching to CPU!")
    # Initialize the pretrained model
    clip_model, clip_preprocess = clip.load(args.clip_type, device=device)
    gan_model = make_gan(gan_type=args.gan_type)
    xmodel = XModel(
        device=device,
        clip_model=clip_model,
        gan_model=gan_model
    )
    print(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--gan_type", type=str, default="biggan")
    parser.add_argument("--clip_type", type=str, default="ViT-B/32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", help='batch size', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--gpu", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    assert args.train is not None or args.test is not None

    if args.train:
        train(args)
    if args.test:
        test(args)