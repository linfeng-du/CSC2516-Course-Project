import torch

from torch import nn
from torch.optim import Adam

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

        # Freeze the pretrained components
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.gan_model.parameters():
            param.requires_grad = False

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

        # Initialize the optimizer and loss function
        params = list(self.mu_map_layer.parameters()) + \
                list(self.log_sigma_map_layer.parameters())
        self.optimizer = Adam(params, lr=args.lr)
        self.loss_fn = nn.MSELoss()

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

    def forward(self, tokenized_prompts):
        z_t = self.get_text_latent_feature(tokenized_prompts)
        z_t.requires_grad = True
        z_mu = self.mu_map_layer(z_t)  # [B, H']
        z_log_sigma = self.log_sigma_map_layer(z_t)  # [B, H']
        z_sigma = torch.exp(z_log_sigma)
        eps = torch.randn(z_mu.shape).to(self.device)
        z_tilde = z_mu + z_sigma * eps
        images = self.gan_model(z_tilde)  # [B, I, I]
        return images, z_t
