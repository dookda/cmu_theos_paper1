"""
===========================================================
SegFormer และ UPerNet สำหรับ Land Cover Segmentation
===========================================================
สองตัวนี้เติมช่องว่างของชุดสถาปัตยกรรมเดิม ซึ่งเป็น CNN encoder-decoder
ยุค 2015–2020 ทั้งหมด (U-Net, U-Net++, DeepLabV3+, HRNet) ไม่มี transformer เลย

- SegFormer (MiT-B2) : encoder เป็น hierarchical transformer, decoder เป็น MLP ล้วน
                       ตัวแทนของสายที่สร้างบริบทระยะไกลด้วย attention
- UPerNet (ConvNeXt) : decoder แบบ pyramid pooling + FPN บน CNN ยุคใหม่
                       ทำหน้าที่เป็นตัวคุม เพื่อแยกว่าผลต่างมาจาก "transformer"
                       หรือมาจาก "สถาปัตยกรรมยุคใหม่" เฉย ๆ

ทั้งคู่เป็นของ segmentation_models_pytorch จึงเข้า training loop เดิมได้ทันที
และ weights ของ mit_b2 ถูก cache ไว้บน Erawan แล้วจากงาน paper2
"""

import segmentation_models_pytorch as smp


def create_segformer(
    encoder_name: str = "mit_b2",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
):
    """สร้าง SegFormer

    Parameters
    ----------
    encoder_name : str - mit_b0 ถึง mit_b5 (paper2 ใช้ mit_b2 กับข้อมูลชุดเดียวกัน)
    encoder_weights : str - "imagenet" หรือ None
    in_channels : int - 3 (RGB) หรือ 4 (RGB+NIR)
    num_classes : int - จำนวนคลาส
    """
    return smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


def create_upernet(
    encoder_name: str = "tu-convnext_tiny",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
):
    """สร้าง UPerNet

    หมายเหตุสำคัญ: หัวของ UPerNet มีสาขา pyramid pooling ที่ยุบเหลือ 1x1
    ทำให้ BatchNorm พังถ้า batch เหลือ 1 ตอน train (`Expected more than 1 value
    per channel`) ฝั่ง dataloader จึงต้องกันไม่ให้ batch สุดท้ายเหลือใบเดียว
    ตอน eval ไม่มีปัญหาเพราะ BatchNorm อยู่ในโหมด running stats

    encoder ใช้ timm ผ่าน prefix `tu-` เช่น tu-convnext_tiny, tu-convnext_small
    """
    return smp.UPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
