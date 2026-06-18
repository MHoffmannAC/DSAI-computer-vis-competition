import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
            if fpath.is_file() and fpath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                yield fpath, idx


def load_metadata_lookup() -> dict:
    """Loads CSV and builds a quick lookup mapping filename -> handedness string"""
    csv_path = Path(TEST_IMAGE_DIR) / "metadata.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        # Ensure uniform casing and whitespace stripping
        df['filename'] = df['filename'].astype(str).str.strip()
        df['handedness'] = df['handedness'].astype(str).str.strip().str.capitalize() # "Left" or "Right"
        return dict(zip(df['filename'], df['handedness']))
    except Exception:
        return {}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--apply_preprocess", choices=["True", "False"], required=True)
    parser.add_argument("--flip", choices=["True", "False"], required=True)
    parser.add_argument("--rotate", choices=["0", "90", "180", "270"], required=True)
    parser.add_argument("--zoom", type=str, default="normal", choices=["normal", "in", "out"])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    apply_preprocess = (args.apply_preprocess == "True")
    should_flip = (args.flip == "True")
    rotate_deg = int(args.rotate)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    model = None

    try:
        if args.model_type == "Custom":
            model = tf.keras.models.load_model(args.model_path)
        else:
            model = tf.keras.models.load_model(
                args.model_path,
                custom_objects={
                    "preprocess_input": ALLOWED_MODELS[args.model_type]["family"].preprocess_input,
                },
            )

        input_shape = model.input_shape
        input_size = (input_shape[1], input_shape[2])

        # 1. Load the lookup dictionary here
        metadata_lookup = load_metadata_lookup()

        paths = list(iter_test_image_paths())
        results_by_file = {}
        correct_preds = 0
        total_preds = 0

        for fpath, label in paths:
            img = Image.open(fpath).convert("RGB")
            filename_key = fpath.name

            # Look up native hand value from CSV metadata mapping (defaults to "Unknown" if file isn't listed)
            native_hand = metadata_lookup.get(filename_key, "Unknown")

            # 1. Flip applied FIRST to switch handedness properties
            if should_flip:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # 2. Rotations applied SECOND
            if rotate_deg == 90:
                img = img.transpose(Image.Transpose.ROTATE_90)
            elif rotate_deg == 180:
                img = img.transpose(Image.Transpose.ROTATE_180)
            elif rotate_deg == 270:
                img = img.transpose(Image.Transpose.ROTATE_270)

            original_width, original_height = img.size

            if args.zoom == "in":
                crop_fraction = 0.80
                left = int((1 - crop_fraction) * original_width / 2)
                top = int((1 - crop_fraction) * original_height / 2)
                right = int((1 + crop_fraction) * original_width / 2)
                bottom = int((1 + crop_fraction) * original_height / 2)

                img = img.crop((left, top, right, bottom))

            elif args.zoom == "out":
                scale_fraction = 0.80
                new_w = int(original_width * scale_fraction)
                new_h = int(original_height * scale_fraction)

                resized_hand = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                padded_img = Image.new("RGB", (original_width, original_height), (128, 128, 128))

                paste_x = (original_width - new_w) // 2
                paste_y = (original_height - new_h) // 2
                padded_img.paste(resized_hand, (paste_x, paste_y))

                img = padded_img

            # Preprocessing & Prediction Execution Matrix
            arr = np.array(img.resize(input_size)).astype("float32")
            if apply_preprocess:
                if args.model_type == "Custom":
                    arr /= 255.0
                else:
                    arr = ALLOWED_MODELS[args.model_type]["family"].preprocess_input(arr)

            batch = np.expand_dims(arr, axis=0)
            preds = model.predict(batch, verbose=0)
            pred_class = int(np.argmax(preds[0]))

            is_correct = (pred_class == label)
            if is_correct:
                correct_preds += 1
            total_preds += 1

            # 2. Save native handedness property directly into the individual file results record
            results_by_file[filename_key] = {
                "y_true": int(label),
                "y_pred": pred_class,
                "correct": is_correct,
                "native_handedness": native_hand  # <-- Appended tracking parameter
            }

        # Package output payload mapping matrix variables
        output_payload = {
            "flip": should_flip,
            "rotate": rotate_deg,
            "overall_accuracy": correct_preds / total_preds if total_preds > 0 else 0.0,
            "predictions": results_by_file
        }

        with open(args.output_json, "w") as f:
            json.dump(output_payload, f)

        tf.keras.backend.clear_session()
        gc.collect()

    finally:
        if model is not None:
            model = None
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()
