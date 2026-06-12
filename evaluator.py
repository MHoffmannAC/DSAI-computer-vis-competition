import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import (
    densenet,
    efficientnet,
    efficientnet_v2,
#    inception_resnet_v2,
    inception_v3,
    mobilenet,
    mobilenet_v2,
    mobilenet_v3,
    nasnet,
    resnet,
    resnet_v2,
    vgg16,
    vgg19,
    xception,
)

ALLOWED_MODELS = {
    "Custom": {"family": "custom", "label": "Custom"},
    "DenseNet121": {"family": densenet, "label": "DenseNet121"},
    "DenseNet169": {"family": densenet, "label": "DenseNet169"},
    "DenseNet201": {"family": densenet, "label": "DenseNet201"},
    "EfficientNetB0": {"family": efficientnet, "label": "EfficientNetB0"},
    "EfficientNetB1": {"family": efficientnet, "label": "EfficientNetB1"},
    "EfficientNetB2": {"family": efficientnet, "label": "EfficientNetB2"},
    "EfficientNetB3": {"family": efficientnet, "label": "EfficientNetB3"},
    "EfficientNetB4": {"family": efficientnet, "label": "EfficientNetB4"},
#    "EfficientNetB5": {"family": efficientnet, "label": "EfficientNetB5"},
#    "EfficientNetB6": {"family": efficientnet, "label": "EfficientNetB6"},
    "EfficientNetV2B0": {"family": efficientnet_v2, "label": "EfficientNetV2B0"},
    "EfficientNetV2B1": {"family": efficientnet_v2, "label": "EfficientNetV2B1"},
    "EfficientNetV2B2": {"family": efficientnet_v2, "label": "EfficientNetV2B2"},
    "EfficientNetV2B3": {"family": efficientnet_v2, "label": "EfficientNetV2B3"},
    "EfficientNetV2S": {"family": efficientnet_v2, "label": "EfficientNetV2S"},
#    "EfficientNetV2M": {"family": efficientnet_v2, "label": "EfficientNetV2M"},
#    "InceptionResNetV2": {"family": inception_resnet_v2, "label": "InceptionResNetV2"},
    "InceptionV3": {"family": inception_v3, "label": "InceptionV3"},
    "MobileNet": {"family": mobilenet, "label": "MobileNet"},
    "MobileNetV2": {"family": mobilenet_v2, "label": "MobileNetV2"},
    "MobileNetV3Small": {"family": mobilenet_v3, "label": "MobileNetV3Small"},
    "MobileNetV3Large": {"family": mobilenet_v3, "label": "MobileNetV3Large"},
    "NASNetMobile": {"family": nasnet, "label": "NASNetMobile"},
    "ResNet50": {"family": resnet, "label": "ResNet50"},
#    "ResNet101": {"family": resnet, "label": "ResNet101"},
#    "ResNet152": {"family": resnet, "label": "ResNet152"},
    "ResNet50V2": {"family": resnet_v2, "label": "ResNet50V2"},
#    "ResNet101V2": {"family": resnet_v2, "label": "ResNet101V2"},
#    "ResNet152V2": {"family": resnet_v2, "label": "ResNet152V2"},
    "VGG16": {"family": vgg16, "label": "VGG16"},
    "VGG19": {"family": vgg19, "label": "VGG19"},
    "Xception": {"family": xception, "label": "Xception"},
}

# ==== CONFIGURATION & CONSTANTS ====
TEST_IMAGE_DIR = "test_images"
CLASS_NAMES = ["A", "B", "C"]

def iter_test_image_paths():
    base_dir = Path(TEST_IMAGE_DIR)

    for idx, cls in enumerate(CLASS_NAMES):
        folder = base_dir / cls
        if not folder.exists():
            continue

        for fpath in sorted(folder.iterdir()):
            if fpath.is_file():
                yield fpath, idx


def generate_augmented_images(img: Image.Image):
    flip_variants = [
        img,
        #img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        #img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        #img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
        #    Image.Transpose.FLIP_TOP_BOTTOM
        #),
    ]

    def zoom_in(im, factor=1.1):
        w, h = im.size
        new_w, new_h = int(w / factor), int(h / factor)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = im.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BICUBIC)

    def zoom_out(im, factor=0.9):
        w, h = im.size
        new_w, new_h = int(w * factor), int(h * factor)
        resized = im.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (w, h))
        canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        return canvas

    for fimg in flip_variants:
        yield fimg
        #yield zoom_in(fimg)
        #yield zoom_out(fimg)


def write_progress(progress_file, done, total, accuracy):
    with open(progress_file, "w") as f:
        json.dump(
            {
                "done": done,
                "total": total,
                "progress": done / total,
                "accuracy": accuracy,
            },
            f,
        )

def evaluate_model_streaming(
    model: tf.keras.Model,
    input_size: tuple[int, int],
    model_type: str,
    apply_preprocess: bool,
    progress_file,
):
    paths = list(iter_test_image_paths())
    total = len(paths)

    y_true = []
    y_pred = []

    correct = 0
    total_preds = 0

    current_acc = 0

    try:

        for i, (fpath, label) in enumerate(paths):
            img = Image.open(fpath).convert("RGB")

            batch = []
            for aug_img in generate_augmented_images(img):
                arr = np.array(aug_img.resize(input_size)).astype("float32")

                if apply_preprocess:
                    if model_type == "Custom":
                        arr /= 255.0
                    else:
                        arr = ALLOWED_MODELS[model_type]["family"].preprocess_input(arr)

                batch.append(arr)

            batch = np.stack(batch)  # shape (12, H, W, 3)

            preds = model.predict(batch, verbose=0)

            del batch

            for p in preds:
                pred_class = np.argmax(p)

                y_true.append(label)
                y_pred.append(pred_class)

                if pred_class == label:
                    correct += 1

                total_preds += 1

            del preds

            current_acc = correct / total_preds

            write_progress(
                progress_file,
                (i + 1) * 12,
                total * 12,
                current_acc,
            )

    finally:
        gc.collect()

    return current_acc, np.array(y_pred), np.array(y_true)


def main():

    model = None
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    apply_preprocess = sys.argv[3] == "True"
    output_json = sys.argv[4]
    progress_json = sys.argv[5]

    try:
        if model_type == "Custom":
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "preprocess_input": ALLOWED_MODELS[model_type]["family"].preprocess_input,
                },
            )

        input_shape = model.input_shape
        input_size = (input_shape[1], input_shape[2])

        acc, y_pred, y_true = evaluate_model_streaming(
            model,
            input_size,
            model_type,
            apply_preprocess,
            progress_json,
        )

        result = {
            "accuracy": float(acc),
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
        }

        with open(output_json, "w") as f:
            json.dump(result, f)

    finally:
        if model is not None:
            model = None
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()
