"""DINOv3-SAT ViT-L/16 with a light multi-scale head, for Paper 1 addendum E.

Backbone: `vit_large_patch16_dinov3.sat493m` (timm) — DINOv3 ViT-L distilled from the
7B model pretrained on SAT-493M, 493 M Maxar RGB tiles at 0.6 m. Weights come from a
local copy (compute nodes have no route out):
  $P1_DINOV3_WEIGHTS, default /project/gisnorth/paper1/weights/vit_large_patch16_dinov3_sat493m.pth

Head (fixed before any run, not tuned): features of blocks 6, 12, 18 and 24 (all at
stride 16) are re-assembled into a stride 4/8/16/32 pyramid as in ViTDet's simple
feature pyramid, fused top-down FPN-style to 256 channels, and upsampled to full
resolution. The model exposes `backbone` so the trainer can give it a lower learning
rate.

The Paper 1 dataset standardises with ImageNet mean/std; the wrapper converts back to
0-1 and applies the SAT-493M statistics the backbone was trained with.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = "/project/gisnorth/paper1/weights/vit_large_patch16_dinov3_sat493m.pth"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LAYERS = [5, 11, 17, 23]
C = 256


def _cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class DinoV3Segmenter(nn.Module):
    def __init__(self, num_classes, load_weights=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=False,
                                          dynamic_img_size=True)
        if load_weights:
            sd = torch.load(os.environ.get("P1_DINOV3_WEIGHTS", DEFAULT_WEIGHTS), map_location="cpu")
            self.backbone.load_state_dict(sd)
        cfg = self.backbone.pretrained_cfg
        d = self.backbone.embed_dim
        self.reassemble = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(d, d // 2, 2, 2), nn.GELU(),
                          nn.ConvTranspose2d(d // 2, C, 2, 2)),            # stride 4
            nn.ConvTranspose2d(d, C, 2, 2),                                 # stride 8
            nn.Conv2d(d, C, 1),                                             # stride 16
            nn.Conv2d(d, C, 3, stride=2, padding=1),                        # stride 32
        ])
        self.smooth = nn.ModuleList([_cbr(C, C) for _ in LAYERS])
        self.fuse = _cbr(C * len(LAYERS), C)
        self.cls = nn.Conv2d(C, num_classes, 1)
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("sat_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("sat_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = (x * self.in_std + self.in_mean).clamp(0, 1)
        x = (x - self.sat_mean) / self.sat_std
        feats = self.backbone.forward_intermediates(x, indices=LAYERS, intermediates_only=True)
        pyr = [r(f) for r, f in zip(self.reassemble, feats)]
        for i in range(len(pyr) - 2, -1, -1):                              # top-down
            pyr[i] = pyr[i] + F.interpolate(pyr[i + 1], size=pyr[i].shape[-2:],
                                            mode="bilinear", align_corners=False)
        pyr = [s(p) for s, p in zip(self.smooth, pyr)]
        size = pyr[0].shape[-2:]
        out = self.fuse(torch.cat([pyr[0]] + [F.interpolate(p, size=size, mode="bilinear",
                                                            align_corners=False)
                                              for p in pyr[1:]], 1))
        return F.interpolate(self.cls(out), size=(h, w), mode="bilinear", align_corners=False)


def create_dinov3(num_classes=7, in_channels=3, load_weights=True):
    if in_channels != 3:
        raise ValueError("DINOv3-SAT is an RGB model; use the rgb conditions")
    return DinoV3Segmenter(num_classes, load_weights=load_weights)
