import argparse
import torch
import clip
from xmodel import XModel
from dataloader import get_dataloader
from pytorch_pretrained_gans import make_gan
from tqdm import tqdm


def train_for_one_epoch(xmodel, train_dl, args):
    for image, text, image_id, caption_id in tqdm(train_dl):
        # print(image, text, image_id, caption_id)
        pass

def eval_model(model, val_dl, args):
    pass


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

    # Initialize the model to train
    xmodel = XModel(
        device,
        clip_model,
        gan_model,
        args
    )

    # Get the dataloader
    train_dl = get_dataloader("train", xmodel, args)
    val_dl = get_dataloader("test", xmodel, args)

    # Start training
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train_for_one_epoch(xmodel, train_dl, args)
        eval_model(xmodel, val_dl, args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--gan_type", type=str, default="biggan")
    parser.add_argument("--clip_type", type=str, default="ViT-B/32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", help='batch size', type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--dataset_name", type=str, default="C-CUB")
    parser.add_argument("--comp_type", type=str, default="color")
    parser.add_argument("--tokenize_ds", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    if args.train:
        train(args)
    if args.test:
        test(args)
