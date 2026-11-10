import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import common
from qwen_finetune import LORA_DIR
from qwen_zeroshot import MODEL_NAME

MERGED_DIR = common.PROJECT_DIR / "qwen_merged"
LLAMA_CPP_DIR = common.PROJECT_DIR / "llama.cpp"
LM_STUDIO_DIR = Path.home() / ".lmstudio" / "models" / "local" / "qwen2vl-fer-finetuned"


def merge_lora():
    # float32 on the CPU: adding the small LoRA update to the weights in bfloat16 would round part of it away
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, LORA_DIR).merge_and_unload()
    model.save_pretrained(MERGED_DIR)
    AutoProcessor.from_pretrained(MODEL_NAME).save_pretrained(MERGED_DIR)
    print(f"merged model saved in {MERGED_DIR}")


def convert(converter, outfile, outtype, *flags):
    command = [sys.executable, str(converter), str(MERGED_DIR), "--outfile", str(outfile), "--outtype", outtype, *flags]
    subprocess.run(command, check=True)
    print(f"saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cpp", type=Path, default=LLAMA_CPP_DIR, help="folder of the llama.cpp repository")
    parser.add_argument("--out", type=Path, default=LM_STUDIO_DIR, help="folder LM Studio reads the model from")
    parser.add_argument("--keep-merged", action="store_true", help="keep the 9 GB merged model in qwen_merged/")
    args = parser.parse_args()

    converter = args.llama_cpp / "convert_hf_to_gguf.py"
    if not converter.exists():
        sys.exit(f"{converter} is missing. Run: git clone --depth 1 https://github.com/ggml-org/llama.cpp")
    if not (LORA_DIR / "adapter_model.safetensors").exists():
        sys.exit(f"No adapter in {LORA_DIR}. Run qwen_finetune.py first.")

    merge_lora()
    args.out.mkdir(parents=True, exist_ok=True)
    convert(converter, args.out / "qwen2vl_fer-q8_0.gguf", "q8_0")
    # LM Studio pairs the model with the vision file in the same folder whose name starts with "mmproj"
    convert(converter, args.out / "mmproj-qwen2vl_fer-f16.gguf", "f16", "--mmproj")
    if not args.keep_merged:
        shutil.rmtree(MERGED_DIR)


if __name__ == "__main__":
    main()
