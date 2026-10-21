"""SatlasPretrain Aerial Swin-B (+FPN) for Paper 1 addendum E.

Remote-sensing-pretrained baseline: Swin-B pretrained on 0.5-2 m aerial imagery
(SatlasPretrain `Aerial_SwinB_SI`), the closest public pretraining domain to THEOS-2 at
0.5 m. Same construction as Paper 2's `train_unet_rspretrained.py` (FPN features, light
conv head at full resolution) so the two papers use one implementation.

Satlas aerial models expect RGB in 0-1. The Paper 1 dataset standardises with ImageNet
mean/std, so the wrapper undoes that first; the data pipeline stays identical to every
other architecture.

Weights come from a local file (compute nodes have no route out):
  $P1_SATLAS_WEIGHTS, default /project/gisnorth/paper2/weights/aerial_swinb_si.pth
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = "/project/gisnorth/paper2/weights/aerial_swinb_si.pth"
# addendum H2: same Swin-V2-B + FPN + head, ImageNet-1k weights (torchvision IMAGENET1K_V1)
IMAGENET_WEIGHTS = "/project/gisnorth/paper1/weights/swin_v2_b_imagenet1k_v1.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SatlasSegmenter(nn.Module):
    def __init__(self, num_classes, weights_path=None, load_weights=True, imagenet=False):
        super().__init__()
        from satlaspretrain_models.model import Model
        from satlaspretrain_models.utils import Backbone
        self.imagenet = imagenet
        w = None
        if load_weights and not imagenet:
            w = torch.load(weights_path or os.environ.get("P1_SATLAS_WEIGHTS", DEFAULT_WEIGHTS),
                           map_location="cpu")
        self.backbone = Model(num_channels=3, multi_image=False, backbone=Backbone.SWINB,
                              fpn=True, head=None, num_categories=None, weights=w)
        if load_weights and imagenet:
            sd = torch.load(os.environ.get("P1_SWINV2B_IMAGENET", IMAGENET_WEIGHTS), map_location="cpu")
            self.backbone.backbone.backbone.load_state_dict(sd, strict=True)
        with torch.no_grad():
            ch = self.backbone(torch.zeros(1, 3, 256, 256))[0].shape[1]
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True), nn.Conv2d(128, num_classes, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if not self.imagenet:
            x = (x * self.std + self.mean).clamp(0, 1)   # back to the 0-1 range Satlas expects
        out = self.head(self.backbone(x)[0])
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def create_satlas(num_classes=7, in_channels=3, load_weights=True, encoder="aerial_swinb_si"):
    if in_channels != 3:
        raise ValueError("Satlas Aerial_SwinB_SI is an RGB model; use the rgb conditions")
    if encoder not in ("aerial_swinb_si", "imagenet_swinv2b"):
        raise ValueError(f"unknown satlas encoder {encoder}")
    return SatlasSegmenter(num_classes, load_weights=load_weights,
                           imagenet=encoder == "imagenet_swinv2b")
