import torch
import torch.optim as optim
import torch.nn as nn
from components.gan_loss import GANLoss
from components.generator_net import GeneratorNet
from components.patch_discriminator import PatchDiscriminator


class Model(nn.Module):
    def __init__(self, device, lambda_L1=100.):
        super().__init__()

        self.device = device
        self.lambda_L1 = lambda_L1

        self.generator_net = GeneratorNet(1, 2).to(self.device)
        self.discriminator_net = PatchDiscriminator(3).to(self.device)
        self.generator_loss = GANLoss(mode='vanilla').to(self.device)
        self.discriminator_loss = nn.L1Loss()
        self.generator_optimizer = optim.Adam(self.generator_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_ab = self.generator_net(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_preds = self.discriminator_net(fake_image.detach())
        self.loss_D_fake = self.generator_loss(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.discriminator_net(real_image)
        self.loss_D_real = self.generator_loss(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_preds = self.discriminator_net(fake_image)
        self.loss_G_GAN = self.generator_loss(fake_preds, True)
        self.loss_G_L1 = self.discriminator_loss(self.fake_ab, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.discriminator_net.train()
        self.set_requires_grad(self.discriminator_net, True)
        self.discriminator_optimizer.zero_grad()
        self.backward_D()
        self.discriminator_optimizer.step()

        self.generator_net.train()
        self.set_requires_grad(self.discriminator_net, False)
        self.generator_optimizer.zero_grad()
        self.backward_G()
        self.generator_optimizer.step()
