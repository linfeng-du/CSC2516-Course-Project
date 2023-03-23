import argparse
import torch
import clip
import torch.nn.functional as F
from xmodel import XModel
from dataloader import get_dataloader
from pytorch_pretrained_gans import make_gan
from tqdm import tqdm


def train_for_one_epoch(xmodel, train_dl, args):
    train_loss = 0.0
    for images, tokenized_prompts, image_ids, caption_ids in tqdm(train_dl):

        xmodel.optimizer.zero_grad()

        gan_images, z_t = xmodel(tokenized_prompts)
        z_i = xmodel.get_image_latent_feature(images)

        loss = xmodel.loss_fn(z_t, z_i)
        loss.backward()
        xmodel.optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl.dataset)
    return train_loss


def eval_model(model, val_dl, args):
    pass

def save_model(xmodel, args):
    save_info = {
        'mu_map_layer': xmodel.mu_map_layer.state_dict(),
        'log_sigma_map_layer': xmodel.log_sigma_map_layer.state_dict(),
        'optimizer': xmodel.optimizer.state_dict(),
        'clip_type': args.clip_type,
        'gan_type': args.gan_type
    }

    torch.save(save_info, args.train)
    print(f"save the model to {args.train}")

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
    max_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train_loss = train_for_one_epoch(xmodel, train_dl, args)
        print(f"Train loss: {train_loss}")
        if train_loss < max_loss:
            save_model(xmodel, args)
            max_loss = train_loss
        eval_model(xmodel, val_dl, args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="../model/test_model.pt")
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
    parser.add_argument("--tokenize_ds", type=bool, default="True")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    if args.train:
        train(args)
    if args.test:
        test(args)
