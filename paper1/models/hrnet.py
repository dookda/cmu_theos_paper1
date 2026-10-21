"""
===========================================================
HRNet Model สำหรับ Land Cover Segmentation
===========================================================
HRNet (High-Resolution Network) เป็นโมเดลที่:
- รักษา high-resolution representations ตลอดทั้งเครือข่าย
- มี multi-resolution parallel streams
- ให้ผลดีกับงานที่ต้องการความละเอียดสูง เช่น satellite imagery
- เหมาะสำหรับ land cover mapping ที่มีขอบเขตซับซ้อน
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_hrnet(
    encoder_name: str = "tu-hrnet_w48",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
):
    """
    สร้างโมเดล HRNet สำหรับ Semantic Segmentation

    Parameters:
    -----------
    encoder_name : str - ชื่อ HRNet backbone
        รองรับ: "tu-hrnet_w18", "tu-hrnet_w32", "tu-hrnet_w48"
        - w18: เล็กสุด, เร็วสุด (~9M params)
        - w32: กลาง (~29M params)
        - w48: ใหญ่สุด, แม่นยำสุด (~65M params)
    encoder_weights : str - "imagenet" สำหรับ pretrained weights
    in_channels : int - จำนวน input channels
    num_classes : int - จำนวน classes

    Returns:
    --------
    model : nn.Module - โมเดล HRNet + FPN decoder
    """
    # ใช้ FPN decoder ร่วมกับ HRNet encoder
    # FPN (Feature Pyramid Network) ช่วยรวม features จากหลาย scale
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    return model



def get_model_info(model, name="HRNet"):
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
    for variant in ["tu-hrnet_w18", "tu-hrnet_w32", "tu-hrnet_w48"]:
        print("=" * 50)
        try:
            model = create_hrnet(encoder_name=variant, num_classes=7)
            get_model_info(model, f"HRNet ({variant})")

            dummy_input = torch.randn(2, 3, 512, 512)
            output = model(dummy_input)
            print(f"   Input:  {dummy_input.shape}")
            print(f"   Output: {output.shape}")
            print(f"   ✅ ทำงานได้ปกติ!\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
