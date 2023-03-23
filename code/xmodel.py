import torch
# import clip
from torch import nn
# from pytorch_pretrained_gans import make_gan


class XModel(nn.Module):

    def __init__(
            self,
            device,
            clip_model,
            gan_model,
    ):
        super().__init__()
        self.device = device
        self.clip_model = clip_model
        self.gan_model = gan_model
        self.map_layer = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, self.gan_model.dim_z)
        )

    def get_text_latent_feature(self, tokenized_prompts):
        """
        Get the text latent feature vector
        :param tokenized_prompts: B * S Tensor
        :return: text latent feature vector [B * H]
        """
        text_latent_feature = self.model.encode_text(tokenized_prompts)
        return text_latent_feature

    def get_iamge_latent_feature(self, images):
        """
        Get the image latent feature vector
        :param images: B * I * I
        :return: image latent feature vector [B * H]
        """
        text_latent_feature = self.model.encode_image(images)
        return text_latent_feature

    def forward(self, tokenized_prompts):
        z_t = self.get_text_latent_feature(tokenized_prompts)
        z_tilde = self.map_layer(z_t)  # [B, H']
        images = self.gan_model(z_tilde)  # [B, I, I]
        return images, z_t