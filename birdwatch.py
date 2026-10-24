#!/usr/bin/env python3
"""
Bird watcher.
Stage 1: SSD MobileNet (COCO) finds "bird" boxes in the camera frame.
Stage 2: iNaturalist MobileNet classifies the cropped bird into a species.
Saves crop + species + timestamp to ~/birds and logs to detections.csv.
"""
import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # fallback if you installed full TensorFlow / ai-edge-litert
    from ai_edge_litert.interpreter import Interpreter

# ---------------- configuration ----------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
OUT_DIR = APP_DIR / "birds"

DETECTOR_MODEL = MODEL_DIR / "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
DETECTOR_LABELS = MODEL_DIR / "coco_labels.txt"
CLASSIFIER_MODEL = MODEL_DIR / "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
CLASSIFIER_LABELS = MODEL_DIR / "inat_bird_labels.txt"

DETECT_THRESHOLD = 0.50     # min confidence that "something is a bird"
SPECIES_THRESHOLD = 0.30    # below this the file is saved as "unknown"
CROP_PADDING = 0.15         # extra margin around the bird box
COOLDOWN_SECONDS = 30       # don't re-save the same species more often than this
LOOP_DELAY = 0.3            # seconds between frames
FRAME_SIZE = (1640, 1232)   # full-ish res for good crops
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
MODEL_FILES = (
    DETECTOR_MODEL.name,
    DETECTOR_LABELS.name,
    CLASSIFIER_MODEL.name,
    CLASSIFIER_LABELS.name,
)
# -----------------------------------------------


def load_interpreter(path):
    interp = Interpreter(model_path=str(path), num_threads=4)
    interp.allocate_tensors()
    return interp


def load_coco_bird_id(path):
    """Find the bird class id in numbered or plain COCO labels."""
    for index, line in enumerate(path.read_text().splitlines()):
        m = re.match(r"\s*(\d+)\s+(.+)", line)
        if m:
            if m.group(2).strip().lower() == "bird":
                return int(m.group(1))
        elif line.strip().lower() == "bird":
            return index
    raise RuntimeError("'bird' not found in coco labels")


def load_labels(path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def validate_model_files(model_dir):
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing model files in {model_dir}: {files}. "
            "Place the models and labels there, or pass --model-dir."
        )


def run_detector(interp, img):
    """Return list of (score, (left, top, right, bottom)) in pixel coords for birds."""
    inp = interp.get_input_details()[0]
    h, w = inp["shape"][1:3]
    small = img.resize((w, h))
    interp.set_tensor(inp["index"], np.expand_dims(np.asarray(small, dtype=np.uint8), 0))
    interp.invoke()
    out = interp.get_output_details()
    boxes = interp.get_tensor(out[0]["index"])[0]
    classes = interp.get_tensor(out[1]["index"])[0]
    scores = interp.get_tensor(out[2]["index"])[0]
    count = int(interp.get_tensor(out[3]["index"])[0])

    results = []
    W, H = img.size
    for i in range(count):
        if int(classes[i]) == BIRD_ID and scores[i] >= DETECT_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            results.append((float(scores[i]),
                            (xmin * W, ymin * H, xmax * W, ymax * H)))
    return results


def pad_box(box, size, pad):
    l, t, r, b = box
    W, H = size
    pw, ph = (r - l) * pad, (b - t) * pad
    return (max(0, int(l - pw)), max(0, int(t - ph)),
            min(W, int(r + pw)), min(H, int(b + ph)))


def classify_species(interp, labels, crop):
    inp = interp.get_input_details()[0]
    h, w = inp["shape"][1:3]
    arr = np.asarray(crop.resize((w, h)), dtype=np.uint8)
    interp.set_tensor(inp["index"], np.expand_dims(arr, 0))
    interp.invoke()
    out = interp.get_output_details()[0]
    scores = interp.get_tensor(out["index"])[0].astype(np.float32)
    scale, zero = out["quantization"]
    if scale:  # dequantize uint8 -> float confidence
        scores = (scores - zero) * scale
    idx = int(np.argmax(scores))
    return labels[idx], float(scores[idx])


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def process_frame(frame,
                  f,
                  detector,
                  classifier,
                  species_labels,
                  last_seen,
                  writer,
                  out_dir=OUT_DIR):
    img = Image.fromarray(frame)
    for det_score, box in run_detector(detector, img):
        crop = img.crop(pad_box(box, img.size, CROP_PADDING))
        species, conf = classify_species(
            classifier, species_labels, crop)
        if conf < SPECIES_THRESHOLD:
            species = "unknown"

        now = datetime.now()
        if now.timestamp() - last_seen.get(species, 0) < COOLDOWN_SECONDS:
            continue
        last_seen[species] = now.timestamp()

        day_dir = out_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        name = f"{now:%H%M%S}_{safe_name(species)}_{conf:.2f}.jpg"
        path = day_dir / name
        crop.save(path, quality=92)
        writer.writerow([now.isoformat(timespec="seconds"),
                         species, f"{conf:.2f}",
                         f"{det_score:.2f}", str(path)])
        f.flush()
        print(f"{now:%H:%M:%S}  {species} ({conf:.0%})  -> {path}")


def camera_detect_loop(model_dir=MODEL_DIR, out_dir=OUT_DIR):
    from picamera2 import Picamera2

    validate_model_files(model_dir)
    global BIRD_ID
    BIRD_ID = load_coco_bird_id(model_dir / DETECTOR_LABELS.name)
    detector = load_interpreter(model_dir / DETECTOR_MODEL.name)
    classifier = load_interpreter(model_dir / CLASSIFIER_MODEL.name)
    species_labels = load_labels(model_dir / CLASSIFIER_LABELS.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "detections.csv"
    new_log = not log_path.exists()
    last_seen = {}

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}))
    cam.start()
    time.sleep(2)  # let auto-exposure settle
    print("Bird watcher running. Ctrl+C to stop.")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["timestamp", "species", "species_conf",
                             "detect_conf", "file"])
        try:
            while True:
                frame = cam.capture_array()          # numpy HxWx3
                process_frame(frame, f, detector, classifier, species_labels,
                              last_seen, writer, out_dir)
                time.sleep(LOOP_DELAY)
        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()


def folder_detect_loop(img_folder, model_dir=MODEL_DIR, out_dir=OUT_DIR):
    if not img_folder.is_dir():
        raise FileNotFoundError(f"Image input directory does not exist: {img_folder}")

    validate_model_files(model_dir)
    global BIRD_ID
    BIRD_ID = load_coco_bird_id(model_dir / DETECTOR_LABELS.name)
    detector = load_interpreter(model_dir / DETECTOR_MODEL.name)
    classifier = load_interpreter(model_dir / CLASSIFIER_MODEL.name)
    species_labels = load_labels(model_dir / CLASSIFIER_LABELS.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "detections.csv"
    new_log = not log_path.exists()
    last_seen = {}

    images = sorted(
        path for path in img_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No supported images found in {img_folder}.")
        return
    print(f"Processing {len(images)} images in {img_folder}.")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["timestamp", "species", "species_conf",
                             "detect_conf", "file"])
        for img_path in images:
            with Image.open(img_path) as image:
                frame = np.asarray(image.convert("RGB"))
            process_frame(frame, f, detector, classifier, species_labels,
                          last_seen, writer, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Detect and classify birds in images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=APP_DIR / "birds" / "to_process",
        help="folder of images to process (default: ./birds/to_process)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"folder containing the TFLite models and labels (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help=f"folder for detections and output crops (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="capture from a Raspberry Pi camera instead of processing local images",
    )
    args = parser.parse_args()

    if args.camera:
        camera_detect_loop(args.model_dir, args.output_dir)
    else:
        folder_detect_loop(args.input_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
