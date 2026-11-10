# Facial expression classification on FER+

Four models compared on the same images: an MLP, a CNN, Qwen2-VL-2B zero-shot, and Qwen2-VL-2B fine-tuned with LoRA.

## Setup

1. Install Python 3.10 or newer.
2. Install PyTorch for your graphics card: the CUDA build from pytorch.org for NVIDIA, or the ROCm build for AMD. Nothing in this project installs or changes PyTorch.
3. The FER+ dataset is included: `train/`, `validation/` and `test/` each hold one sub-folder per emotion (angry, contempt, disgust, fear, happy, neutral, sad, suprise) with PNG images.
4. `internet_images/` holds eight extra pictures from the internet, one per emotion. Their sources and licenses are in `internet_images/SOURCES.md`.

## Run everything

Double-click `run_all.bat`, or type:

```
python run_all.py
```

It does these steps one after another, never two at the same time:

1. Checks that PyTorch can see the graphics card.
2. Installs any library from `requirements.txt` that is missing.
3. Downloads the Qwen model (about 4.4 GB, only the first time).
4. Loads the dataset with `common.py`.
5. Runs `mlp.py`, `cnn.py`, `qwen_zeroshot.py` and `qwen_finetune.py`.
6. Runs `analysis.ipynb` to make the comparison table and the charts.

Everything printed is also saved in `results/<step>.log`, and `results/run_all.log` lists which steps worked and how long each took. The computer is kept awake while the script runs.

A full run takes about 8 to 9 hours on an RX 7800 XT, nearly all of it the four epochs of Qwen fine-tuning. For a rehearsal of about 10 minutes first:

```
python run_all.py --n-train 2000 --n-test 200 --epochs 1
```

To run only some of the models, name them: `python run_all.py --models mlp cnn`. Each script can also be run by itself, for example `python cnn.py`.

## What a run produces

The MLP and the CNN train for 80 epochs and Qwen fine-tuning for 4. Each of the three keeps the epoch that did best on validation images. Every model is then scored on the same 3,573 test images, predicts the eight internet pictures, and writes `results/<model>.json`. The MLP and the CNN also save their trained weights as `results/mlp.pt` and `results/cnn.pt`, and the Qwen adapter is saved in `qwen_lora/`.

The notebook writes `results/summary.csv` and the chart images, including `results/internet_images.png`.

Use no flags for the final numbers, and run all four models on one machine so the timings are comparable.
