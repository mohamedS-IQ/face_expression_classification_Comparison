import argparse
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS = ["mlp", "cnn", "qwen_zeroshot", "qwen_finetune"]
QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
CHECK_PYTORCH = (
    "import torch, torchvision; "
    "assert torch.cuda.is_available(), 'PyTorch cannot see a graphics card'; "
    "print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
)
DOWNLOAD_QWEN = f"from huggingface_hub import snapshot_download; print('Qwen is stored in', snapshot_download('{QWEN_MODEL}'))"
ES_CONTINUOUS_AND_SYSTEM_REQUIRED = 0x80000001


def keep_computer_awake():
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS_AND_SYSTEM_REQUIRED)


def run(name, command):
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"\n===== {name} started {datetime.now():%Y-%m-%d %H:%M:%S} =====", flush=True)
    start = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(RESULTS_DIR / f"{name}.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    minutes = (time.perf_counter() - start) / 60
    print(f"===== {name} finished with exit code {code} after {minutes:.1f} minutes =====", flush=True)
    return code, minutes


def setup(models):
    python = sys.executable
    steps = [
        ("setup_pytorch", [python, "-c", CHECK_PYTORCH], "Install PyTorch for your graphics card first (see README.md)."),
        ("setup_packages", [python, "-m", "pip", "install", "-r", "requirements.txt"], "Check the internet connection."),
    ]
    if any(name.startswith("qwen") for name in models):
        steps.append(("setup_qwen_download", [python, "-c", DOWNLOAD_QWEN], "Check the internet connection and the free disk space."))
    steps.append(("setup_data", [python, "common.py"], "Check that the train, validation, test and internet_images folders are here."))
    for name, command, advice in steps:
        code, _ = run(name, command)
        if code != 0:
            sys.exit(f"\nThe step {name} failed, so nothing was trained. {advice} The details are in results/{name}.log.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="which models to run, in this order")
    args, model_args = parser.parse_known_args()
    sys.stdout.reconfigure(errors="replace")
    keep_computer_awake()
    setup(args.models)

    summary = []
    for name in args.models:
        summary.append((name, *run(name, [sys.executable, f"{name}.py", *model_args])))
    notebook = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", "analysis.ipynb"]
    summary.append(("analysis", *run("analysis", notebook)))

    lines = [f"{name:15s} {'ok' if code == 0 else 'FAILED':8s} {minutes:8.1f} minutes" for name, code, minutes in summary]
    (RESULTS_DIR / "run_all.log").write_text("\n".join(lines) + "\n")
    print("\n===== summary =====\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
