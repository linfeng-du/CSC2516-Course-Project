import torch
import clip
from torch import nn
from pytorch_pretrained_gans import make_gan

class XModel(nn.Module):

    def __init__(
            self,
            device,
            clip_model,
            gan_model,
    ):
        self.device = device
        self.clip_model = clip_model
        self.gan_model = gan_model

    def get_text_latent_feature(self, tokenized_prompt):
        """
        Get the text latent feature vector
        :param tokenized_prompt: B * S Tensor
        :return: text latent feature vector [B * H]
        """
        text_latent_feature = self.model.encode_text(tokenized_prompt)
        return text_latent_feature

    def get_iamge_latent_feature(self, images):
        """
        Get the image latent feature vector
        :param images: B * I * I
        :return: image latent feature vector [B * H]
        """
        text_latent_feature = self.model.encode_image(images)
        return text_latent_feature


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
gan_model = make_gan(gan_type='biggan')
model = XModel(
    device=device,
    clip_model=clip_model,
    gan_model=gan_model
)
