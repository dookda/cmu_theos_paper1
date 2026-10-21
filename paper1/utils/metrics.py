"""
===========================================================
Metrics สำหรับประเมินผล Semantic Segmentation
===========================================================
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    คำนวณ metrics สำหรับ semantic segmentation:
    - Overall Accuracy (OA)
    - Mean IoU (mIoU) 
    - Per-class IoU
    - F1-Score (Dice coefficient)
    - Kappa coefficient
    """

    def __init__(self, num_classes: int, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [
            f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """รีเซ็ต confusion matrix"""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        อัพเดท confusion matrix

        Parameters:
        -----------
        pred : torch.Tensor - prediction (B, C, H, W) หรือ (B, H, W)
        target : torch.Tensor - ground truth (B, H, W)
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # สร้าง mask สำหรับค่าที่ valid
        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        # อัพเดท confusion matrix
        for t, p in zip(target, pred):
            self.confusion_matrix[int(t), int(p)] += 1

    def get_overall_accuracy(self):
        """Overall Accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-8)

    def get_per_class_iou(self):
        """IoU สำหรับแต่ละ class"""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-8)
        return iou

    def get_mean_iou(self):
        """Mean IoU (mIoU)"""
        iou = self.get_per_class_iou()
        return np.nanmean(iou)

    def get_per_class_f1(self):
        """F1-Score สำหรับแต่ละ class"""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    def get_kappa(self):
        """Cohen's Kappa coefficient"""
        total = self.confusion_matrix.sum()
        po = np.diag(self.confusion_matrix).sum() / total
        pe = np.sum(
            self.confusion_matrix.sum(
                axis=0) * self.confusion_matrix.sum(axis=1)
        ) / (total * total)
        kappa = (po - pe) / (1 - pe + 1e-8)
        return kappa

    def get_summary(self):
        """สรุปผลทั้งหมด"""
        oa = self.get_overall_accuracy()
        miou = self.get_mean_iou()
        per_class_iou = self.get_per_class_iou()
        per_class_f1 = self.get_per_class_f1()
        kappa = self.get_kappa()

        summary = {
            "overall_accuracy": oa,
            "mean_iou": miou,
            "kappa": kappa,
            "per_class_iou": {
                name: iou for name, iou in zip(self.class_names, per_class_iou)
            },
            "per_class_f1": {
                name: f1 for name, f1 in zip(self.class_names, per_class_f1)
            },
        }

        return summary

    def print_summary(self):
        """แสดงผลสรุป"""
        summary = self.get_summary()

        print(f"\n{'='*50}")
        print(f"  📊 ผลการประเมิน Segmentation")
        print(f"{'='*50}")
        print(f"  Overall Accuracy: {summary['overall_accuracy']:.4f}")
        print(f"  Mean IoU (mIoU):  {summary['mean_iou']:.4f}")
        print(f"  Kappa:            {summary['kappa']:.4f}")

        print(f"\n  {'Class':<20} {'IoU':<10} {'F1':<10}")
        print(f"  {'-'*40}")
        for name in self.class_names:
            iou = summary['per_class_iou'][name]
            f1 = summary['per_class_f1'][name]
            print(f"  {name:<20} {iou:.4f}     {f1:.4f}")

        print(f"{'='*50}")

        return summary
