#!/usr/bin/env python3
"""
Raspberry Pi bird watcher.
Stage 1: SSD MobileNet (COCO) finds "bird" boxes in the camera frame.
Stage 2: iNaturalist MobileNet classifies the cropped bird into a species.
Buster / legacy-camera version (picamera, Python 3.7).
Saves detected bird crops and records to an SQLite database in ~/birds.
"""
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import io
import picamera

from bird_data import initialize_database, record_detection
from tflite_runtime.interpreter import Interpreter

# ---------------- configuration ----------------
MODEL_DIR = Path.home() / "birdwatch" / "models"
OUT_DIR = Path.home() / "birds"

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
# -----------------------------------------------


def load_interpreter(path):
    interp = Interpreter(model_path=str(path), num_threads=4)
    interp.allocate_tensors()
    return interp


def load_coco_bird_id(path):
    """Find the class id for 'bird'. Handles '<id> <name>' lines and plain name lines."""
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        m = re.match(r"(\d+)\s+(.+)", line)
        if m:
            idx, name = int(m.group(1)), m.group(2).strip()
        else:
            idx, name = i, line
        if name.lower() == "bird":
            return idx
    raise RuntimeError("'bird' not found in coco labels, check the file contents")


def load_labels(path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


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


def main():
    global BIRD_ID
    BIRD_ID = load_coco_bird_id(DETECTOR_LABELS)
    detector = load_interpreter(DETECTOR_MODEL)
    classifier = load_interpreter(CLASSIFIER_MODEL)
    species_labels = load_labels(CLASSIFIER_LABELS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    database_path = OUT_DIR / "detections.sqlite3"
    initialize_database(
        database_path,
        legacy_csv_path=OUT_DIR / "detections.csv",
        output_dir=OUT_DIR,
    )
    last_seen = {}

    cam = picamera.PiCamera(resolution=FRAME_SIZE)
    time.sleep(2)  # let auto-exposure settle
    buf = io.BytesIO()
    print("Bird watcher running. Ctrl+C to stop.")

    try:
        while True:
            buf.seek(0)
            buf.truncate()
            cam.capture(buf, format="jpeg")
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
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

                day_dir = OUT_DIR / now.strftime("%Y-%m-%d")
                day_dir.mkdir(exist_ok=True)
                name = f"{now:%H%M%S}_{safe_name(species)}_{conf:.2f}.jpg"
                path = day_dir / name
                crop.save(path, quality=92)
                record_detection(
                    database_path,
                    now.isoformat(timespec="seconds"),
                    species,
                    conf,
                    det_score,
                    path.relative_to(OUT_DIR).as_posix(),
                )
                print(f"{now:%H:%M:%S}  {species} ({conf:.0%})  -> {path}")
            time.sleep(LOOP_DELAY)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
