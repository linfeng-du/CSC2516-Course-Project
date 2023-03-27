import torch

from torch import nn
from torch.optim import Adam
from torchvision.transforms import Resize

class XModel(nn.Module):

    def __init__(
            self,
            device,
            clip_model,
            gan_model,
            args
    ):
        super().__init__()
        self.device = device
        self.clip_model = clip_model
        self.gan_model = gan_model
        self.image_size = self.clip_model.visual.input_resolution
        self.args = args

        # Initialize the map layer
        self.mu_map_layer = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, self.gan_model.dim_z)
        )
        self.log_sigma_map_layer = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, self.gan_model.dim_z)
        )

        # Resize layer for GAN-generated image
        self.resize = Resize(self.image_size)

        # Initialize the optimizer and loss function
        params = list(self.mu_map_layer.parameters()) + \
                list(self.log_sigma_map_layer.parameters())
        if self.args.finetune_gan:
            params += list(self.gan_model.parameters())
        self.optimizer = Adam(params, lr=args.lr)

    def get_text_latent_feature(self, tokenized_prompts):
        """
        Get the text latent feature vector
        :param tokenized_prompts: B * S Tensor
        :return: text latent feature vector [B * H]
        """
        text_latent_feature = self.clip_model.encode_text(tokenized_prompts)
        return text_latent_feature

    def get_image_latent_feature(self, images):
        """
        Get the image latent feature vector
        :param images: B * I * I
        :return: image latent feature vector [B * H]
        """
        text_latent_feature = self.clip_model.encode_image(images)
        return text_latent_feature

    def kl_divergence(self, mu, sigma):
        """
        Get the kl divergence respect to standard gaussian
        :param mu: B * O
        :param sigma: B * O
        :return: kl: scalar
        """
        kl = -0.5 * torch.sum(1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2), dim=-1).mean()
        return kl * self.args.kl

    def loss_fn(self, gan_image_eb, text_eb, mu, sigma):
        """
        Get the loss
        :param gan_image_eb: B * EB
        :param text_eb: B * EB
        :param mu: B * O
        :param sigma: B * O
        :return: loss: scalar
        """
        cosine_sim = nn.CosineSimilarity(dim=-1)
        loss = -cosine_sim(gan_image_eb, text_eb).mean() + self.kl_divergence(mu, sigma)
        return loss

    def forward(self, tokenized_prompts):
        z_t = self.get_text_latent_feature(tokenized_prompts).float()
        # z_t.requires_grad = True
        z_mu = self.mu_map_layer(z_t)  # [B, H']
        z_log_sigma = self.log_sigma_map_layer(z_t)  # [B, H']
        z_sigma = torch.exp(z_log_sigma)
        eps = torch.randn(z_mu.shape).to(self.device)
        z_tilde = z_mu + z_sigma * eps
        images = self.gan_model(z_tilde)  # [B, I, I]
        return images, z_t, z_mu, z_sigma
