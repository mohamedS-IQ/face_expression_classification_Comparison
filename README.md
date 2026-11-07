# Facial expression classification on FER+

Four models compared on the same images: an MLP, a CNN, Qwen2-VL-2B zero-shot, and Qwen2-VL-2B fine-tuned with LoRA.

## Setup

1. Install Python 3.10 or newer.
2. Install PyTorch with CUDA from pytorch.org, then `pip install -r requirements.txt`.
3. The FER+ dataset is included: `train/`, `validation/` and `test/` each hold one sub-folder per emotion (angry, contempt, disgust, fear, happy, neutral, sad, suprise) with PNG images.

## Run

```
python common.py
python mlp.py
python cnn.py
python qwen_zeroshot.py
python qwen_finetune.py
```

Each script scores its model on the same 3,573 test images and writes `results/<model>.json`. Then open `analysis.ipynb` and run all cells to get the comparison table and charts in `results/`.

All scripts accept `--n-train N`, `--n-test N` and `--epochs N` for quick tests; use no flags for the final numbers, and run all four on one machine so the timings are comparable.
