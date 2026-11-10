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

A full run takes about 2 to 3 hours on an RX 7800 XT, most of it Qwen. For a rehearsal of about 10 minutes first:

```
python run_all.py --n-train 2000 --n-test 200 --epochs 1
```

To run only some of the models, name them: `python run_all.py --models mlp cnn`. Each script can also be run by itself, for example `python cnn.py`.

## What a run produces

The MLP and the CNN train for 30 epochs and Qwen fine-tuning for 1. Each of the three keeps the epoch that did best on validation images. The MLP and the CNN also save their trained weights as `results/mlp.pt` and `results/cnn.pt`, and the Qwen adapter is saved in `qwen_lora/`.

Every model is then scored separately on four sets of pictures, the same pictures for every model:

| Set | Pictures | What it shows |
|---|---|---|
| train | the first 10,000 training pictures | how well the model knows the pictures it learned from |
| validation | all 8,341 validation pictures | the pictures used to choose the best epoch |
| test | all 3,573 test pictures | the honest score, on pictures that played no part in training |
| internet | the 8 internet pictures | a demo only, far too few to measure a model |

For each set, `results/<model>.json` holds the number of pictures, the accuracy, the macro F1, the F1 of each emotion, the confusion matrix and the model's answer for every picture. A training accuracy far above the validation and test accuracy means the model memorized its training pictures.

The notebook writes `results/summary.csv` (one row per model, one accuracy column per set) and the chart images, including `results/quality.png` and `results/internet_images.png`.

Use no flags for the final numbers, and run all four models on one machine so the timings are comparable.

## Use the fine-tuned Qwen in LM Studio

LM Studio reads GGUF files, so the LoRA adapter in `qwen_lora/` has to be merged into Qwen and converted first:

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp
python export_gguf.py
```

This writes the model and its vision file (`mmproj-...`) to `~/.lmstudio/models/local/qwen2vl-fer-finetuned/`. It needs about 9 GB of free disk space while it runs.

To check the converted model, load it in LM Studio, start the local server (`lms server start`), then:

```
python lmstudio_test.py
```

It scores the first 200 test images through LM Studio. The model was trained on one question, so ask it exactly that: the `PROMPT` in `qwen_zeroshot.py`, with a 224 x 224 picture of a face.
