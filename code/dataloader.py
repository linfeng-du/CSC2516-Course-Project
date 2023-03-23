from dataset import *
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch
from tqdm import tqdm

# # Get data loader
# imsize = 224
# image_transform = transforms.Compose([
#     transforms.Resize((imsize, imsize)),
#     transforms.RandomCrop(imsize),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()]
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )
# data_dir = "../data/"
# dataset_name = "C-CUB"
# comp_type = "color"
# split = "test"
# tokenize = True
# if dataset_name == "C-CUB":
#     images_txt_path = os.path.join(data_dir, dataset_name, "images.txt")
#     bbox_txt_path = os.path.join(data_dir, dataset_name, "bounding_boxes.txt")
#     dataset = CCUBDataset(
#         data_dir,
#         dataset_name,
#         comp_type,
#         split,
#         image_transform,
#         tokenize,
#         images_txt_path,
#         bbox_txt_path
#     )
# elif dataset_name == "C-Flowers":
#     class_id_txt_path = os.path.join(data_dir, "C-Flowers", "class_ids.txt")
#     dataset = CFlowersDataset(
#         data_dir,
#         dataset_name,
#         comp_type,
#         split,
#         image_transform,
#         tokenize,
#         class_id_txt_path
#     )
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True, shuffle=True)
#
# for image, text, image_id, caption_id in tqdm(dataloader):
#     save_image(image[0, ...].data, f"../gen/example_dataset.png")
#     # print(caps)
#     pass
# Get data loader


def get_dataloader(split, model, args):

    # Retrieve the parameters
    imsize = model.image_size
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    comp_type = args.comp_type
    tokenize = args.tokenize_ds

    # Transform the image
    image_transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load the data to dataset
    if dataset_name == "C-CUB":
        images_txt_path = os.path.join(data_dir, dataset_name, "images.txt")
        bbox_txt_path = os.path.join(data_dir, dataset_name, "bounding_boxes.txt")
        dataset = CCUBDataset(
            data_dir,
            dataset_name,
            comp_type,
            split,
            image_transform,
            tokenize,
            images_txt_path,
            bbox_txt_path
        )
    elif dataset_name == "C-Flowers":
        class_id_txt_path = os.path.join(data_dir, "C-Flowers", "class_ids.txt")
        dataset = CFlowersDataset(
            data_dir,
            dataset_name,
            comp_type,
            split,
            image_transform,
            tokenize,
            class_id_txt_path
        )

    # Wrap the dataset into dataloader and return
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True, shuffle=True)
    return dataloader
