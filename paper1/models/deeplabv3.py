"""
===========================================================
DeepLabV3+ Model สำหรับ Land Cover Segmentation
===========================================================
DeepLabV3+ เป็นโมเดล Semantic Segmentation ที่ใช้:
- Atrous Spatial Pyramid Pooling (ASPP) สำหรับจับ multi-scale context
- Encoder-Decoder structure สำหรับรายละเอียดของขอบ
- เหมาะสำหรับการจำแนกพื้นที่ขนาดใหญ่ เช่น land cover mapping
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_deeplabv3(
    encoder_name: str = "resnet101",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
):
    """
    สร้างโมเดล DeepLabV3+

    Parameters:
    -----------
    encoder_name : str - ชื่อ backbone (resnet50, resnet101, efficientnet-b4, ฯลฯ)
    encoder_weights : str - "imagenet" สำหรับ pretrained weights
    in_channels : int - จำนวน input channels (3=RGB, 4=RGBNIR)
    num_classes : int - จำนวน classes ที่ต้องการจำแนก

    Returns:
    --------
    model : nn.Module - โมเดล DeepLabV3+
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # ใช้ raw logits สำหรับ CrossEntropyLoss
    )

    return model


def get_model_info(model):
    """แสดงข้อมูลของโมเดล"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"🏗️  DeepLabV3+ Model Info:")
    print(
        f"   Encoder: {model.encoder.name if hasattr(model.encoder, 'name') else 'N/A'}")
    print(f"   พารามิเตอร์ทั้งหมด: {total_params:,}")
    print(f"   พารามิเตอร์ที่ train ได้: {trainable_params:,}")
    print(f"   ขนาดโมเดล: ~{total_params * 4 / 1e6:.1f} MB (FP32)")

    return total_params, trainable_params


if __name__ == "__main__":
    # ทดสอบสร้างโมเดล
    model = create_deeplabv3(
        encoder_name="resnet101",
        num_classes=7,
        in_channels=3,
    )
    get_model_info(model)

    # ทดสอบ forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"\n   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ โมเดลทำงานได้ปกติ!")
