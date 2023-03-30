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
import os
import json
import random
import pickle
from dataset import *


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def imagenet_classes():
    with open("../data/imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = ["an image of " + class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    return idx2label, cls2label


def get_imagenet_label(args, xmodel, z_t, image_ids, imagenet_label_dict):
    xmodel.eval()
    labels = torch.zeros((len(image_ids))).int().to(z_t.device)
    idx2label, cls2label = None, None
    for i in range(len(image_ids)):
        if imagenet_label_dict.get(image_ids[i]):
            label_list = imagenet_label_dict.get(image_ids[i])
            labels[i] = random.choice(label_list)
        else:
            if idx2label is None and cls2label is None:
                idx2label, cls2label = imagenet_classes()
            else:
                clip_scores = None
                for idx in range(100):
                    label = clip.tokenize(idx2label[idx * 10: idx * 10 + 10]).to(z_t.device)
                    imagenet_label_feat = xmodel.clip_model.encode_text(label).float()
                    imagenet_label_feat = imagenet_label_feat.unsqueeze(0)
                    clip_score = torch.nn.functional.cosine_similarity(z_t[i].unsqueeze(0).unsqueeze(0), imagenet_label_feat, dim=-1)
                    if clip_scores is None:
                        clip_scores = clip_score
                    else:
                        clip_scores = torch.cat((clip_scores, clip_score), dim=-1)
                val, indx = torch.topk(clip_scores, k=1000, dim=-1)
                imagenet_label_dict[image_ids[i]] = indx.tolist()
                label_list = imagenet_label_dict.get(image_ids[i])
                labels[i] = random.choice(label_list)[0]
                # save_dict(imagenet_label_dict, "../data/C-CUB/imagenet_labels.pkl")
                # imagenet_label_dict = load_dict("../data/C-CUB/imagenet_labels.pkl")
                # pass

    if args.one_hot_label:
        labels = torch.eye(1000, dtype=torch.float, device=z_t.device)[labels]
    return labels


def select_imagenet_label(args, xmodel, val_dl, imagenet_label_dict):
    for images, tokenized_prompts, image_ids, caption_ids in tqdm(val_dl):
        tokenized_prompts = tokenized_prompts.to(xmodel.device)
        z_t = xmodel.get_text_latent_feature(tokenized_prompts).float()
        labels = get_imagenet_label(args, xmodel, z_t, image_ids, imagenet_label_dict)
    save_dict(imagenet_label_dict, "../data/C-CUB/imagenet_labels.pkl")


def save_dict(dict, save_path):
    f = open(save_path, 'wb')
    pickle.dump(dict, f)
    f.close()


def load_dict(save_path):
    f = open(save_path, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--condition", action='store_true')
    parser.add_argument("--one_hot_label", action='store_true', help="set true when using biggan")
    parser.add_argument("--load_path", type=str, default="../checkpoint/C-CUB/bigbigan/max_images_200.pt")
    parser.add_argument("--gen_dir", type=str, default="../gen/")
    parser.add_argument("--plot_dir", type=str, default="../plot/")
    parser.add_argument("--save_dir", type=str, default="../checkpoint/")
    parser.add_argument("--gan_type", type=str, default="studiogan")
    parser.add_argument("--gan_model_name", type=str, default="SAGAN")
    parser.add_argument("--clip_type", type=str, default="ViT-B/32")
    parser.add_argument("--max_images", type=int, default=2)
    parser.add_argument("--k", type=int, default=10, help="top k imagenet labels to select from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", help='batch size', type=int, default=64)
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

args = get_args()
seed_everything(seed=args.seed)
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
data_dir = "../data/"
dataset_name = "C-CUB"
comp_type = "color"
split = "test"
tokenize = True
# torch.manual_seed(42)
images_txt_path = os.path.join(data_dir, dataset_name, "images.txt")
bbox_txt_path = os.path.join(data_dir, dataset_name, "bounding_boxes.txt")
dataset = CCUBDataset(
    data_dir,
    dataset_name,
    comp_type,
    split,
    clip_preprocess,
    tokenize,
    images_txt_path,
    bbox_txt_path
)
val_dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)

# Start training
clip_scores = torch.zeros((args.max_images, args.batch_size * len(val_dl)))
imagenet_label_dict = {}
with torch.no_grad():
    select_imagenet_label(args, xmodel, val_dl, imagenet_label_dict)
