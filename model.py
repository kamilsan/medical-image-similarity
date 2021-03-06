import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ReidentificationModel(nn.Module):
    def __init__(self, embedding_size):
        super(ReidentificationModel, self).__init__()

        self.backbone = torchvision.models.resnet18(pretrained=False)
        fc_in_features = self.backbone.fc.in_features

        self.backbone = torch.nn.Sequential(
            *(list(self.backbone.children())[:-1]))
        self.embedding = nn.Linear(fc_in_features, embedding_size)

    def forward_single(self, input):
        x = self.backbone(input)
        x = x.view(x.size()[0], -1)
        emb = self.embedding(x)
        return F.normalize(emb)

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_single(anchor)
        output_positive = self.forward_single(positive)
        output_negative = self.forward_single(negative)

        return output_anchor, output_positive, output_negative
