from src.data import build_transforms


def test_build_transforms_resizes_to_square():
    transform = build_transforms(256)
    from PIL import Image
    img = Image.new("RGB", (640, 480), color="white")
    tensor = transform(img)
    assert tensor.shape == (3, 256, 256)
