import argparse
import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from xmodel import XModel
from dataloader import get_dataloader
from pytorch_pretrained_gans import make_gan
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_imagenet_label(args, xmodel, image_ids, imagenet_label_dict):
    xmodel.eval()
    labels = torch.zeros((len(image_ids))).int().to(xmodel.device)
    for i in range(len(image_ids)):
        label_list = imagenet_label_dict.get(image_ids[i])
        if label_list is None:
            labels[i] = torch.randint(0, 1000, (1,))
        else:
            label_list = label_list[0][:args.k]
            labels[i] = random.choice(label_list)

    if args.one_hot_label:
        labels = torch.eye(1000, dtype=torch.float, device=xmodel.device)[labels.long()]
    return labels


def random_select_one_image(args, xmodel, val_dl, max_clip_score, imagenet_label_dict):
    clip_scores = None
    cosine_sim = nn.CosineSimilarity(dim=-1)
    substitute = 0
    for images, tokenized_prompts, image_ids, caption_ids in tqdm(val_dl):
        tokenized_prompts = tokenized_prompts.to(xmodel.device)
        z_t = xmodel.get_text_latent_feature(tokenized_prompts).float()
        z = xmodel.gan_model.sample_latent(batch_size=val_dl.batch_size).to(xmodel.device)
        if args.condition:
            labels = get_imagenet_label(args, xmodel, image_ids, imagenet_label_dict)
            gan_images = xmodel.gan_model(z=z, y=labels)
        else:
            gan_images = xmodel.gan_model(z)
        gan_images = xmodel.resize(gan_images)
        z_i = xmodel.get_image_latent_feature(gan_images)

        clip_score = cosine_sim(z_i, z_t)
        for i in range(len(image_ids)):
            if max_clip_score.get(image_ids[i]):
                if clip_score[i] > max_clip_score.get(image_ids[i]):
                    save_image(args, gan_images[i, ...], image_ids[i])
                    max_clip_score[image_ids[i]] = clip_score[i].cpu().item()
                    substitute += 1
            else:
                max_clip_score[image_ids[i]] = clip_score[i].cpu().item()
        if clip_scores is None:
            clip_scores = clip_score
        else:
            clip_scores = torch.cat((clip_scores, clip_score))

    return clip_scores.cpu(), substitute / clip_scores.shape[0]


def save_image(args, image, image_id):
    gen_folder = os.path.join(args.gen_dir, args.dataset_name, str(args.seed), args.gan_type + "_epoch_" + str(args.max_images))
    if not os.path.exists(gen_folder):
        os.makedirs(gen_folder)
    image_id = image_id.replace("/", " ")
    obj = image.data.detach().cpu().numpy().transpose((1, 2, 0))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
    obj = np.asarray(np.uint8(obj), dtype=np.uint8)
    img = Image.fromarray(obj)
    file_path = os.path.join(gen_folder, f"{image_id}.png")
    img.save(file_path, "png")


def save_tensor(args, clip_scores, epoch=None):
    save_folder = os.path.join(args.save_dir, args.dataset_name, str(args.seed), args.gan_type)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    clip_scores = clip_scores.cpu()
    if epoch is None:
        epoch = args.max_images
    save_path = os.path.join(save_folder, "max_images_" + str(epoch) + ".pt")
    torch.save(clip_scores, save_path)


def plot_clip_nums(args, clip_scores):
    plot_folder = os.path.join(args.plot_dir, args.dataset_name, str(args.seed), args.gan_type + "_epoch_" + str(args.max_images))
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    y = np.mean(clip_scores, axis=-1)
    x = np.arange(1, y.shape[0] + 1)
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Num Images")
    plt.ylabel("Avg. Best Clip Score")
    plot_path = os.path.join(plot_folder, f"avg  best clip score v.s. num images.png")
    fig.savefig(plot_path)


def plot_epoch_clip(args, clip_scores, epoch):
    plot_folder = os.path.join(args.plot_dir, args.dataset_name, str(args.seed), args.gan_type + "_epoch_" + str(args.max_images))
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    x = clip_scores[epoch, ...]
    bins = np.linspace(0.1, 0.5, num=21)
    fig = plt.figure(figsize=(6.4, 6), dpi=100)
    plt.hist(x, bins=bins)
    plt.xticks(bins, rotation=270)
    plt.xlabel("Best Clip Score")
    plt.ylabel("Num Images")
    plt.title(f"Epcoh: {epoch}")
    plot_path = os.path.join(plot_folder, f"epoch {epoch} best clip score.png")
    fig.savefig(plot_path)


def plot_substitute_rate(args, substitutes):
    plot_folder = os.path.join(args.plot_dir, args.dataset_name, str(args.seed), args.gan_type + "_epoch_" + str(args.max_images))
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    substitutes[0] = 1.0
    y = np.asarray(substitutes)
    x = np.arange(1, y.shape[0] + 1)
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Num Images")
    plt.ylabel("Substitute Rate")
    plot_path = os.path.join(plot_folder, f"substitute rate v.s. num images.png")
    fig.savefig(plot_path)


def load_dict(save_path):
    f = open(save_path, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict


def main(args):
    """
    Train a model according to specifications in args.
    """

    # Claim the device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("No GPU found, switching to CPU!")

    # Initialize the pretrained model
    clip_model, clip_preprocess = clip.load(args.clip_type, device=device)
    if args.gan_type == "studiogan":
        gan_model = make_gan(gan_type=args.gan_type, model_name=args.gan_model_name)
        args.gan_type = os.path.join(args.gan_type, args.gan_model_name)
    else:
        gan_model = make_gan(gan_type=args.gan_type)

    # Initialize the model to train
    xmodel = XModel(
        device,
        clip_model,
        gan_model,
        args
    ).to(device)

    # Get the dataloader
    # train_dl = get_dataloader("train", xmodel, args)
    val_dl = get_dataloader("test", xmodel, args, shuffle=False)

    # Start training
    clip_scores = torch.zeros((args.max_images, args.batch_size * len(val_dl)))
    max_clip_score = {}
    imagenet_label_dict = None
    if args.condition:
        args.gan_type += "_conditional" + "_k_" + str(args.k)
        dict_path = os.path.join(args.data_dir, args.dataset_name, "imagenet_labels.pkl")
        imagenet_label_dict = load_dict(dict_path)
    substitutes = []
    with torch.no_grad():
        for epoch in tqdm(range(args.max_images)):
            print(f"Epoch {epoch}")
            clip_score, substitute_rate = random_select_one_image(args, xmodel, val_dl, max_clip_score, imagenet_label_dict)
            if epoch > 0:
                mask = clip_score > clip_scores[epoch - 1]
                clip_scores[epoch, mask] = clip_score[mask]
                mask = clip_score <= clip_scores[epoch - 1]
                clip_scores[epoch, mask] = clip_scores[epoch - 1, mask]
            else:
                clip_scores[epoch, ...] = clip_score
            substitutes.append(substitute_rate)
            print(f"Avg. Best Clip Score: {torch.mean(clip_scores[epoch, ...]).item()}\t || Substitute Rate: {substitute_rate}")
            if epoch % 50 == 0:
                save_tensor(args, clip_scores, epoch=epoch)
    plot_clip_nums(args, clip_scores.numpy())
    for epoch in np.linspace(0, args.max_images - 1, num=10).astype(int):
        plot_epoch_clip(args, clip_scores.numpy(), epoch)
    plot_substitute_rate(args, substitutes)
    save_tensor(args, clip_scores)


def analysis(args):
    # Start testing
    titel_name = args.gan_type
    dict = {}
    list_k = [1, 10, 20, 50, 100, 1000]
    list_seed = [42, 43, 44]
    for k in list_k:
        for seed in list_seed:
            gan_folder = args.gan_type
            if args.gan_type == "studiogan":
                titel_name = args.gan_model_name
                gan_folder = os.path.join(args.gan_type, args.gan_model_name)
            if args.condition:
                gan_folder += "_conditional" + "_k_" + str(k)
            load_folder = os.path.join(args.save_dir, args.dataset_name, str(seed), gan_folder)
            load_file_name = "max_images_" + str(args.max_images) + ".pt"
            load_path = os.path.join(load_folder, load_file_name)
            if dict.get(k) is None:
                dict[k] = torch.load(load_path).unsqueeze(1)
            else:
                dict[k] = torch.cat((dict[k], torch.load(load_path).unsqueeze(1)), dim=1)

    plot_folder = os.path.join(args.plot_dir, args.dataset_name, str(args.seed),
                               args.gan_type)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig = plt.figure()
    plt.xlabel("Num Images")
    plt.ylabel("Avg. Best Clip Score")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i = 0
    for k in list_k:
        clip_scores = dict[k].numpy()
        clip_scores = np.mean(clip_scores, axis=-1)
        avg = np.mean(clip_scores, axis=-1)
        std = np.std(clip_scores, axis=-1)
        y = avg
        for run in range(clip_scores.shape[1]):
            print(f"k = {k}, final best clip score {clip_scores[-1, run]:.6f}")
            for num in range(clip_scores.shape[0]):
                if clip_scores[num, run] > 0.2841796875:
                    print(f"k = {k}, need to select {num + 1} images")
                    break
        x = np.arange(1, y.shape[0] + 1)
        plt.plot(x, y, color=colors[i % len(colors)], label=f"k={k}")
        r1 = avg - std
        r2 = avg + std
        plt.fill_between(x, r1, r2, color=colors[i % len(colors)], alpha=0.2)
        i += 1
    plt.legend()
    plt.title(titel_name)
    plot_path = os.path.join(plot_folder, f"avg  best clip score v.s. num images.png")
    fig.savefig(plot_path)


def get_caption(args):
    # Claim the device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("No GPU found, switching to CPU!")

    # Initialize the pretrained model
    clip_model, clip_preprocess = clip.load(args.clip_type, device=device)
    if args.gan_type == "studiogan":
        gan_model = make_gan(gan_type=args.gan_type, model_name=args.gan_model_name)
        args.gan_type = os.path.join(args.gan_type, args.gan_model_name)
    else:
        gan_model = make_gan(gan_type=args.gan_type)

    # Initialize the model to train
    xmodel = XModel(
        device,
        clip_model,
        gan_model,
        args
    ).to(device)

    # Get the dataloader
    # train_dl = get_dataloader("train", xmodel, args)
    val_dl = get_dataloader("test", xmodel, args, shuffle=False)
    captions = val_dl.dataset.anno_data["153.Philadelphia_Vireo/Philadelphia_Vireo_0038_794759"]
    print(captions)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--condition", action='store_true')
    parser.add_argument("--one_hot_label", action='store_true', help="set true when using one hot label")
    parser.add_argument("--gen_dir", type=str, default="../gen/")
    parser.add_argument("--plot_dir", type=str, default="../plot/")
    parser.add_argument("--save_dir", type=str, default="../checkpoint/")
    parser.add_argument("--gan_type", type=str, default="biggan")
    parser.add_argument("--gan_model_name", type=str, default="SAGAN")
    parser.add_argument("--clip_type", type=str, default="ViT-B/32")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--k", type=int, default=10, help="top k imagenet labels to select from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", help='batch size', type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--dataset_name", type=str, default="C-CUB")
    parser.add_argument("--comp_type", type=str, default="color")
    parser.add_argument("--tokenize_ds", type=bool, default="True")
    parser.add_argument("--kl", type=float, help="hp of kl", default=0.0)
    parser.add_argument("--finetune_gan", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    seed_everything(seed=args.seed)
    # get_caption(args)
    if args.test:
        analysis(args)
    else:
        main(args)
