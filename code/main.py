import argparse
import torch
import clip
import torch.nn.functional as F
from xmodel import XModel
from dataloader import get_dataloader
from pytorch_pretrained_gans import make_gan
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

def train_for_one_epoch(xmodel, train_dl):
    train_loss = 0.0
    for images, tokenized_prompts, image_ids, caption_ids in tqdm(train_dl):

        xmodel.optimizer.zero_grad()
        tokenized_prompts = tokenized_prompts.to(xmodel.device)
        gan_images, z_t, z_mu, z_sigma = xmodel(tokenized_prompts)
        gan_images = xmodel.resize(gan_images)
        z_i = xmodel.get_image_latent_feature(gan_images)

        loss = xmodel.loss_fn(z_i, z_t, z_mu, z_sigma)
        loss.backward()
        xmodel.optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)
    return train_loss


def eval_model(model, val_dl):
    pass

def save_model(xmodel, args):
    save_info = {
        'mu_map_layer': xmodel.mu_map_layer.state_dict(),
        'log_sigma_map_layer': xmodel.log_sigma_map_layer.state_dict(),
        'optimizer': xmodel.optimizer.state_dict(),
        'clip_type': args.clip_type,
        'gan_type': args.gan_type
    }
    if args.finetune_gan:
        save_info['gan_model'] = xmodel.gan_model.state_dict()

    torch.save(save_info, args.model_path)
    print(f"save the model to {args.model_path}")

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
    ).to(device)

    # Get the dataloader
    train_dl = get_dataloader("train", xmodel, args)
    val_dl = get_dataloader("test", xmodel, args)

    # Start training
    max_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train_loss = train_for_one_epoch(xmodel, train_dl)
        print(f"Train loss: {train_loss}")
        if train_loss < max_loss:
            save_model(xmodel, args)
            max_loss = train_loss
        eval_model(xmodel, val_dl)


def test(args):
    """
    Test a model according to specifications in args.
    """

    # Claim the device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("No GPU found, switching to CPU!")

    # Initialize the pretrained model
    clip_model, clip_preprocess = clip.load(args.clip_type, device=device)
    gan_model = make_gan(gan_type=args.gan_type)

    # Initialize and Load the model to test
    xmodel = XModel(
        device,
        clip_model,
        gan_model,
        args
    ).to(device)
    saved = torch.load(args.model_path)
    xmodel.mu_map_layer.load_state_dict(saved['mu_map_layer'])
    xmodel.log_sigma_map_layer.load_state_dict(saved['log_sigma_map_layer'])
    if saved.get('gan_model'):
        xmodel.gan_model.load_state_dict(saved['gan_model'])
    xmodel.eval()

    # Get the dataloader
    test_dl = get_dataloader("test", xmodel, args)

    # Start testing
    cnt = 0
    gen_folder = os.path.join(args.gen_dir, args.dataset_name, args.model_path)
    if not os.path.exists(gen_folder):
        os.makedirs(gen_folder)
    with torch.no_grad():
        for images, tokenized_prompts, image_ids, caption_ids in tqdm(test_dl):
            tokenized_prompts = tokenized_prompts.to(xmodel.device)
            gan_images, z_t, z_mu, z_sigma = xmodel(tokenized_prompts)
            gan_images = xmodel.resize(gan_images)
            for i in range(gan_images.shape[0]):
                image = gan_images[i, ...]
                obj = image.data.detach().cpu().numpy().transpose((1, 2, 0))
                obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
                obj = np.asarray(np.uint8(obj), dtype=np.uint8)
                img = Image.fromarray(obj)
                file_path = os.path.join(gen_folder, f"{cnt}.png")
                img.save(file_path, "png")
                cnt += 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model_path", type=str, default="../model/test_model_ft_gan.pt")
    parser.add_argument("--gen_dir", type=str, default="../gen/")
    parser.add_argument("--gan_type", type=str, default="biggan")
    parser.add_argument("--clip_type", type=str, default="ViT-B/32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", help='batch size', type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--dataset_name", type=str, default="C-CUB")
    parser.add_argument("--comp_type", type=str, default="color")
    parser.add_argument("--tokenize_ds", type=bool, default="True")
    parser.add_argument("--kl", type=float, help="hp of kl", default=1.0)
    parser.add_argument("--finetune_gan", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    if args.train:
        train(args)
        test(args)
    if args.test:
        test(args)
