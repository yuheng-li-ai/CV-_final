from pathlib import Path

from PIL import Image

from cvhw3_scene.object_c import ObjectCPreprocessConfig, preprocess_object_c


def test_preprocess_object_c_writes_masked_image_mask_and_metadata(tmp_path):
    image_path = tmp_path / "input.png"
    image = Image.new("RGB", (20, 20), "white")
    for x in range(6, 14):
        for y in range(5, 16):
            image.putpixel((x, y), (230, 190, 20))
    image.save(image_path)

    result = preprocess_object_c(
        ObjectCPreprocessConfig(
            input_image=image_path,
            output_image=tmp_path / "masked" / "object_c_auto.png",
            output_mask=tmp_path / "masks" / "object_c_auto.png",
            output_metadata=tmp_path / "metadata.json",
            threshold=245,
            padding=2,
            centered_crop=True,
            dry_run=False,
        )
    )

    assert result.bbox == [4, 3, 16, 18]
    assert result.foreground_pixels > 0
    assert result.output_image.exists()
    assert result.output_mask.exists()
    assert result.output_metadata.exists()

    masked = Image.open(result.output_image)
    mask = Image.open(result.output_mask)
    assert masked.mode == "RGBA"
    assert mask.mode == "L"


def test_preprocess_object_c_dry_run_does_not_write_outputs(tmp_path):
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)

    result = preprocess_object_c(
        ObjectCPreprocessConfig(
            input_image=image_path,
            output_image=tmp_path / "masked.png",
            output_mask=tmp_path / "mask.png",
            output_metadata=tmp_path / "metadata.json",
            dry_run=True,
        )
    )

    assert result.dry_run
    assert not result.output_image.exists()
    assert not result.output_mask.exists()
    assert not result.output_metadata.exists()


def test_preprocess_object_c_can_keep_largest_component(tmp_path):
    image_path = tmp_path / "input.png"
    image = Image.new("RGB", (30, 20), "white")
    for x in range(3, 7):
        for y in range(3, 7):
            image.putpixel((x, y), (230, 190, 20))
    for x in range(12, 25):
        for y in range(4, 17):
            image.putpixel((x, y), (230, 190, 20))
    image.save(image_path)

    result = preprocess_object_c(
        ObjectCPreprocessConfig(
            input_image=image_path,
            output_image=tmp_path / "masked.png",
            output_mask=tmp_path / "mask.png",
            output_metadata=tmp_path / "metadata.json",
            threshold=245,
            padding=0,
            centered_crop=False,
            keep_largest_component=True,
            close_kernel=3,
            dry_run=False,
        )
    )

    assert result.output_mask.exists()
    assert result.foreground_pixels >= 12 * 12
