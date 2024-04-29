import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0, mode="vanilla", target="real"):
        super(GANLoss, self).__init__()
        self.register_buffer("real_labels", torch.tensor(real_label))
        self.register_buffer("fake_labels", torch.tensor(fake_label))

        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

    def __call__(self, predictions, target_is_real):
        if target_is_real:
            labels = self.real_labels
        else:
            labels = self.fake_labels
        labels = labels.expand_as(predictions)

        return self.loss(predictions, labels)
