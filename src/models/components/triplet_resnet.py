import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from src.models.components.resnet import resnet50feat, resnet50, resnet50feattrip
from src.models.components.simCLR_resnet import ResNetSimCLR
import torch.nn.functional as F

__all__ = ["TripletResNet50, TripletResNet50Feat"]


class TripletResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(TripletResNet50, self).__init__()

        # Load the ResNet-50 feat model
        self.resnet50 = resnet50(pretrained=pretrained)

        # Remove the final classification layer (the fully connected layer)
        # self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Define a linear layer to get the latent representations
        self.latent_layer = nn.Linear(1000, 768)
        self.fc_classifier = nn.Linear(768, num_classes)  # classification layer

    def forward(self, anchor, positive, negative):
        # Forward pass for the anchor, positive, and negative samples
        anchor_latent = self.latent_layer(self.resnet50(anchor))
        positive_latent = self.latent_layer(self.resnet50(positive))
        negative_latent = self.latent_layer(self.resnet50(negative))

        anchor_output = self.fc_classifier(anchor_latent)
        positive_output = self.fc_classifier(positive_latent)
        netagive_output = self.fc_classifier(negative_latent)

        return (
            F.normalize(anchor_latent, dim=1),
            F.normalize(positive_latent, dim=1),
            F.normalize(negative_latent, dim=1),
            anchor_output,
            positive_output,
            netagive_output,
        )


def triplet_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model for triplet training.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TripletResNet50(pretrained=pretrained)
    return model


class TripletResNet50Feat(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TripletResNet50Feat, self).__init__()

        # Load the ResNet-50 feat model
        self.resnet50 = resnet50feattrip(pretrained=pretrained)

        # Remove the final classification layer (the fully connected layer)
        # self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Define a linear layer to get the latent representations
        # self.latent_layer = nn.Linear(2048, 768)
        # self.fc_classifier = nn.Linear(768, num_classes)  # classification layer

        self.latent_layer = nn.Sequential(
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )
        self.fc_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes),
        )  # classification layer

    def forward(self, anchor, positive, negative):
        # Forward pass for the anchor, positive, and negative samples
        anchor_latent = self.latent_layer(self.resnet50(anchor))
        positive_latent = self.latent_layer(self.resnet50(positive))
        negative_latent = self.latent_layer(self.resnet50(negative))

        anchor_output = self.fc_classifier(anchor_latent)
        positive_output = self.fc_classifier(positive_latent)
        netagive_output = self.fc_classifier(negative_latent)

        return (
            F.normalize(anchor_latent, dim=1),
            F.normalize(positive_latent, dim=1),
            F.normalize(negative_latent, dim=1),
            anchor_output,
            positive_output,
            netagive_output,
        )


def triplet_resnet50feat(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model for triplet training.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TripletResNet50Feat(pretrained=pretrained)
    return model


def simCLR_resnet50feat(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model for triplet training.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSimCLR(base_model="resnet50", out_dim=768)
    return model


# class SupConResNet(nn.Module):
#     """backbone + projection head"""

#     def __init__(self, name="resnet50", head="mlp", feat_dim=128):
#         super(SupConResNet, self).__init__()
#         model_fun, dim_in = model_dict[name]
#         self.encoder = model_fun()
#         if head == "linear":
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == "mlp":
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim),
#             )
#         else:
#             raise NotImplementedError("head not supported: {}".format(head))

#     def forward(self, x):
#         feat = self.encoder(x)
#         feat = F.normalize(self.head(feat), dim=1)
#         return feat
