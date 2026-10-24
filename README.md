# BirdWatchPi

This is a Raspberry PI based project for AI-assisted bird watching.

## Running locally without a Pi camera

`birdwatch.py` processes image files by default and does not require `picamera2`.
Put images in `birds/to_process` (JPEG, PNG, GIF, BMP, or WebP), then run:

```bash
python birdwatch.py
```

Download the model files required by the detector and bird classifier into
`~/birdwatch/models` (the default):

```bash
mkdir -p ~/birdwatch/models
cd ~/birdwatch/models
curl -fLO https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite
curl -fLO https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
curl -fLO https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v2_1.0_224_inat_bird_quant.tflite
curl -fLO https://raw.githubusercontent.com/google-coral/test_data/master/inat_bird_labels.txt
```

Then run `python birdwatch.py`. Use `--model-dir` if you keep the files
elsewhere, or `--input-dir` and `--output-dir` to select other folders.
Camera capture remains available on a Raspberry Pi with `picamera2` installed
by running `python birdwatch.py --camera`.

## Preparations

### Option A (recommended): install a current Raspberry OS

Buster has been end-of-life for a while, so its package repositories may give you trouble, and newer Python wheels won’t install. A fresh Raspberry Pi OS Bookworm (Lite or Desktop) on the Pi 3 takes about 15 minutes. After that, everything in my first walkthrough applies unchanged, using birdwatch.py. If you don’t have anything valuable on the SD card, I’d do this.

### Option B: stay on Buster

Use birdwatch_buster.py, which is identical apart from the camera code (picamera instead of picamera2, and a JPEG capture converted into a PIL image).

1. Enable and test the camera

```bash
sudo raspi-config        # Interface Options → Camera → Enable, then reboot
vcgencmd get_camera      # should print: supported=1 detected=1
raspistill -o test.jpg   # should create a picture
```

2. Install dependencies

```bash
sudo apt update
sudo apt install -y python3-picamera python3-pil python3-numpy python3-venv python3-pip
python3 -m venv --system-site-packages ~/birdwatch/venv
source ~/birdwatch/venv/bin/activate
pip install tflite-runtime
```

Python 3.7 is old, so the last pip command may not find a matching wheel. If it fails, use Google’s package index:

```bash
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```
If apt update itself throws 404 errors, the Buster repos have moved and it’s another sign to go with Option A.

3. Models: same four wget downloads as before (they’re plain TFLite files and work on any version).

4. Run it

```bash
python ~/birdwatch/birdwatch_buster.py
```

5. Autostart: same systemd service as before, with ExecStart pointing at birdwatch_buster.py.

Differences you’ll notice on Buster
Each capture takes around 1 second at full resolution on the Pi 3, so you’ll get roughly one check per second or a little less. That’s still fine for birds.
You can’t use rpicam-* commands, only raspistill / raspivid.
If the script errors on the tflite_runtime import, the install in step 2 didn’t succeed. Send me the error and I’ll help.

Which option do you want to go with? If you tell me your Python version (python3 --version) after trying Option B, I can pin the right tflite-runtime wheel for you.

## Testing

Run tests:

```bash
.venv/bin/python -m unittest tests.test_birdwatch -v
```