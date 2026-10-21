"""
===========================================================
UNet Model สำหรับ Land Cover Segmentation
===========================================================
UNet เป็นโมเดล Semantic Segmentation ที่ใช้:
- Encoder-Decoder architecture แบบสมมาตร
- Skip connections ระหว่าง encoder และ decoder
- เหมาะสำหรับงานที่ต้องการรายละเอียดสูง เช่น การจำแนกขอบเขตที่ดิน
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_unet(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
    decoder_attention: str = "scse",
):
    """
    สร้างโมเดล UNet

    Parameters:
    -----------
    encoder_name : str - ชื่อ backbone (resnet34, resnet50, efficientnet-b3, ฯลฯ)
    encoder_weights : str - "imagenet" สำหรับ pretrained weights
    in_channels : int - จำนวน input channels
    num_classes : int - จำนวน classes
    decoder_attention : str - ประเภท attention ("scse" หรือ None)

    Returns:
    --------
    model : nn.Module - โมเดล UNet
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        decoder_attention_type=decoder_attention,
    )

    return model


def create_unetpp(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
):
    """
    สร้างโมเดล UNet++ (Nested UNet)
    ปรับปรุงจาก UNet ด้วย dense skip connections
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    return model


def get_model_info(model, name="UNet"):
    """แสดงข้อมูลของโมเดล"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"🏗️  {name} Model Info:")
    print(f"   พารามิเตอร์ทั้งหมด: {total_params:,}")
    print(f"   พารามิเตอร์ที่ train ได้: {trainable_params:,}")
    print(f"   ขนาดโมเดล: ~{total_params * 4 / 1e6:.1f} MB (FP32)")

    return total_params, trainable_params


if __name__ == "__main__":
    # ทดสอบ UNet
    print("=" * 50)
    model = create_unet(encoder_name="resnet50", num_classes=7)
    get_model_info(model, "UNet")

    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"   Input:  {dummy_input.shape}")
    print(f"   Output: {output.shape}")

    # ทดสอบ UNet++
    print("\n" + "=" * 50)
    model_pp = create_unetpp(encoder_name="resnet50", num_classes=7)
    get_model_info(model_pp, "UNet++")

    output_pp = model_pp(dummy_input)
    print(f"   Input:  {dummy_input.shape}")
    print(f"   Output: {output_pp.shape}")
    print(f"\n   ✅ โมเดลทำงานได้ปกติ!")
