from PIL import Image, ImageDraw

from src.preprocess import preprocess_catalog, sharpness_score, standardize


def make_sharp_image(size=(300, 200), color="red"):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 10):
        draw.line([(x, 0), (x, size[1])], fill="black", width=1)
    return img


def make_blurry_image(size=(300, 200)):
    return Image.new("RGB", size, "gray")


def test_standardize_pads_to_square():
    img = make_sharp_image((300, 200))
    out = standardize(img, 224)
    assert out.size == (224, 224)


def test_sharpness_score_distinguishes_blur():
    sharp = sharpness_score(make_sharp_image())
    blurry = sharpness_score(make_blurry_image())
    assert sharp > blurry


def test_preprocess_catalog_drops_duplicates_and_blurry():
    sharp = make_sharp_image()
    params = {
        "preprocess": {"image_size": 224, "min_sharpness": 50, "remove_background": False}
    }
    images = [sharp, sharp, make_blurry_image()]
    kept, report = preprocess_catalog(images, params)

    assert report["input"] == 3
    assert report["dropped_duplicate"] == 1
    assert report["dropped_blurry"] == 1
    assert report["kept"] == 1
    assert len(kept) == 1
    assert kept[0].size == (224, 224)
