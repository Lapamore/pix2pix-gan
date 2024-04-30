from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from skimage import color
import matplotlib.pyplot as plt
from model import Model


class ImageColorizer:
    def __init__(self, image_path: str, device: str, save_image: bool):
        self.image_path = image_path
        self.save_image = save_image
        self.device = device

    def load_model(self):
        self.model = Model(self.device).to(self.device)
        self.model.load_state_dict(torch.load("weights/model_weights.pth"))

    def lab_to_rgb(self, L, fake_ab):
        L = (L + 1.0) * 50.0
        fake_ab = fake_ab * 110.0
        image = torch.cat([L, fake_ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        image_rgb = color.lab2rgb(image)
        return image_rgb.squeeze(0)

    def inference_image(self):
        image = Image.open(self.image_path).convert("RGB")
        transform_resize = transforms.Resize((256, 256))
        image_transformed = transform_resize(image)
        image_numpy = np.array(image_transformed)
        image_lab = color.rgb2lab(image_numpy).astype("float32")
        image_tensor = transforms.ToTensor()(image_lab).to(self.device)
        L = image_tensor[[0], ...] / 50.0 - 1.0
        
        self.model.generator_net.eval()
        with torch.no_grad():
            self.model.L = L.unsqueeze(0)
            self.model.forward()

        self.model.generator_net.train()
        fake_ab = self.model.fake_ab.detach()
        color_image_predicted = self.lab_to_rgb(self.model.L, fake_ab)

        plt.imshow(color_image_predicted)
        plt.axis("off")

        if self.save_image:
            plt.savefig(self.image_path.split(".")[0] + "_predicted.png")
