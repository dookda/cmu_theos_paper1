"""
===========================================================
Loss Functions สำหรับ Semantic Segmentation
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss - เหมาะสำหรับข้อมูลที่ class ไม่สมดุล
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth=1.0, num_classes=7, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(
            target.long(), self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        if self.class_weights is not None:
            w = self.class_weights.to(dice.device)
            return 1.0 - (dice * w).sum(dim=1).mean() / w.sum()
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - ลดน้ำหนักของตัวอย่างที่จำแนกได้ง่าย
    เหมาะสำหรับข้อมูลที่ class ไม่สมดุลมาก
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=7, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred, target):
        weight = self.class_weights.to(pred.device) if self.class_weights is not None else None
        ce_loss = F.cross_entropy(pred, target.long(), weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    รวม CrossEntropy + Dice Loss
    ให้ผลดีที่สุดสำหรับ segmentation ในหลายกรณี
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=7, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=w)
        self.dice_loss = DiceLoss(num_classes=num_classes, class_weights=class_weights)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.long())
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class BackgroundMaskedComboLoss(nn.Module):
    """CE + Dice over classes 1..C-1 only; reference pixels of class 0 are ignored

    Paper 1 addendum D: background is a residual bucket (Section 4.6), so the model is
    never asked to predict it. Logit 0 is dropped before the softmax, which keeps the
    7-channel head (and checkpoint format) while giving channel 0 no gradient at all.
    """

    def __init__(self, num_classes=7, ce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.k = num_classes - 1
        self.ce_weight, self.dice_weight, self.smooth = ce_weight, dice_weight, smooth

    def forward(self, pred, target):
        logits = pred[:, 1:]
        target = target.long()
        valid = target > 0
        if not valid.any():
            # an all-background batch carries no signal; CE over zero pixels is NaN
            return pred.sum() * 0.0
        t = torch.where(valid, target - 1, torch.full_like(target, -100))
        ce = F.cross_entropy(logits, t, ignore_index=-100)

        prob = F.softmax(logits, dim=1) * valid.unsqueeze(1)
        onehot = F.one_hot(t.clamp(min=0), self.k).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1)
        inter = (prob * onehot).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(loss_name: str, num_classes: int = 7, class_weights=None):
    """
    เลือก loss function ตามชื่อ

    Parameters:
    -----------
    loss_name : str - "ce", "dice", "focal", "combo"
    num_classes : int - จำนวน classes
    class_weights : list/ndarray - น้ำหนักต่อ class สำหรับข้อมูลไม่สมดุล (optional)
    """
    ce_w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    loss_map = {
        "ce": nn.CrossEntropyLoss(weight=ce_w),
        "dice": DiceLoss(num_classes=num_classes, class_weights=class_weights),
        "focal": FocalLoss(num_classes=num_classes, class_weights=class_weights),
        "combo": CombinedLoss(num_classes=num_classes, class_weights=class_weights),
    }

    if loss_name not in loss_map:
        raise ValueError(
            f"ไม่รู้จัก loss: {loss_name}. เลือกจาก: {list(loss_map.keys())}")

    return loss_map[loss_name]
