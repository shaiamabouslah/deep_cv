"""Segmentation model: U-Net with a pretrained CNN encoder, used to detect
the water surface in a satellite image."""
import segmentation_models_pytorch as smp


def build_segmentation_model(encoder_name="resnet34", encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def load_model(checkpoint_path, device="cpu"):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_segmentation_model(checkpoint["encoder_name"], encoder_weights=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    # Older checkpoints predate threshold tuning on the validation set.
    return model, checkpoint["image_size"], checkpoint.get("threshold", 0.5)
