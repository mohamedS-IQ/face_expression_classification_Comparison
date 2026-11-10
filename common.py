import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FOLDERS = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
EMOTIONS = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SPLITS = {"train": "train", "validation": "val", "test": "test"}
IMAGE_SIZE = 48
TRAIN_SCORE_IMAGES = 10000
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_PATH = PROJECT_DIR / "ferplus_cache.npz"
RESULTS_DIR = PROJECT_DIR / "results"
INTERNET_DIR = PROJECT_DIR / "internet_images"


if torch.version.hip and os.environ.get("DISABLE_MIOPEN") == "1":
    torch.backends.cudnn.enabled = False

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
])


def parse_args(default_epochs=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=None, help="use only the first N training images")
    parser.add_argument("--n-test", type=int, default=None, help="score only the first N images of each set")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.seconds = time.perf_counter() - self.start


def load_image(path):
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.uint8)


def load_split(folder):
    images, labels = [], []
    for label, name in enumerate(FOLDERS):
        for path in sorted((folder / name).glob("*.png")):
            images.append(load_image(path))
            labels.append(label)
    images, labels = np.stack(images), np.array(labels, dtype=np.int64)
    order = np.random.RandomState(0).permutation(len(labels))
    return images[order], labels[order]


def load_data():
    if CACHE_PATH.exists():
        return dict(np.load(CACHE_PATH))
    data = {}
    for folder, name in SPLITS.items():
        data[f"x_{name}"], data[f"y_{name}"] = load_split(PROJECT_DIR / folder)
        print(f"loaded {folder}: {len(data[f'y_{name}'])} images")
    np.savez(CACHE_PATH, **data)
    return data


def load_internet_images():
    images = np.stack([load_image(INTERNET_DIR / f"{name.lower()}.png") for name in EMOTIONS])
    labels = np.arange(len(EMOTIONS), dtype=np.int64)
    return images, labels


def scoring_data(data, args):
    train_x = data["x_train"][:args.n_train][:TRAIN_SCORE_IMAGES]
    train_y = data["y_train"][:args.n_train][:TRAIN_SCORE_IMAGES]
    sets = {"train": (train_x, train_y), "validation": (data["x_val"], data["y_val"]), "test": (data["x_test"], data["y_test"])}
    images = {name: x[:args.n_test] for name, (x, y) in sets.items()}
    labels = {name: y[:args.n_test] for name, (x, y) in sets.items()}
    images["internet"], labels["internet"] = load_internet_images()
    return images, labels


class FaceDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.x = torch.from_numpy(images).float().unsqueeze(1) / 255
        self.y = torch.from_numpy(labels)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        if self.augment:
            x = TRAIN_TRANSFORM(x)
        return x, self.y[index]


def predict(model, dataset, batch_size=512):
    device = get_device()
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size):
            predictions.append(model(x.to(device)).argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def warm_up(model, dataset):
    predict(model, dataset)


def train_model(model, train_set, val_set, epochs, lr=1e-3, batch_size=128):
    device = get_device()
    model.to(device)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        val_acc = (predict(model, val_set) == val_set.y.numpy()).mean()
        print(f"epoch {epoch:3d}   loss {total_loss / len(train_set):.4f}   val accuracy {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"best val accuracy {best_acc:.4f}")
    return model


def evaluate(y_true, y_pred):
    labels = list(range(len(EMOTIONS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {name: float(score) for name, score in zip(EMOTIONS, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def report(name, model, train_time_s, inference_time_s, labels, answers, extra=None):
    results = {
        "model": name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_time_s": round(train_time_s, 2),
        "inference_ms_per_image": round(1000 * inference_time_s / len(labels["test"]), 4),
        "params": count_parameters(model),
        "size_mb": round(model_size_mb(model), 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0,
        **(extra or {}),
    }
    for set_name, y_true in labels.items():
        y_pred = np.asarray(answers[set_name])
        results[set_name] = {"images": len(y_true), **evaluate(y_true, y_pred), "predictions": y_pred.tolist()}
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(name)
    for set_name in labels:
        scores = results[set_name]
        print(f"  {set_name:10s} {scores['images']:6d} images   accuracy {scores['accuracy']:.4f}   macro F1 {scores['macro_f1']:.4f}")
    print(f"  train {results['train_time_s']} s   inference {results['inference_ms_per_image']} ms/image")
    guesses = [EMOTIONS[p] if p >= 0 else "unclear" for p in answers["internet"]]
    pairs = [f"{true} -> {guess}" for true, guess in zip(EMOTIONS, guesses)]
    print("  internet images (true -> predicted): " + ", ".join(pairs))
    print(f"saved {path}")
    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = load_data()
    for name in SPLITS.values():
        counts = np.bincount(data[f"y_{name}"], minlength=len(EMOTIONS))
        print(f"{name}: {len(data[f'y_{name}'])} images, per emotion {dict(zip(EMOTIONS, counts.tolist()))}")
    print(f"internet images: {len(load_internet_images()[0])}")

    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(16, 2.5))
    for label, ax in enumerate(axes):
        index = np.flatnonzero(data["y_train"] == label)[0]
        ax.imshow(data["x_train"][index], cmap="gray")
        ax.set_title(EMOTIONS[label])
        ax.axis("off")
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "preview.png", bbox_inches="tight")
    print(f"saved {RESULTS_DIR / 'preview.png'}")
