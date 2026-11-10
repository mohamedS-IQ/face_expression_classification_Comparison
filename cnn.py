import torch
import torch.nn as nn

import common

EPOCHS = 30
LEARNING_RATE = 1e-3


class NativeBatchNorm2d(nn.BatchNorm2d):
    # MIOpen's BatchNorm training kernel fails to compile on ROCm for Windows; run BN natively, convs stay on MIOpen.
    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(x)


BatchNorm2d = NativeBatchNorm2d if torch.version.hip else nn.BatchNorm2d


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(common.EMOTIONS)),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    args = common.parse_args(default_epochs=EPOCHS)
    common.set_seed()
    data = common.load_data()
    train_set = common.FaceDataset(data["x_train"][:args.n_train], data["y_train"][:args.n_train], augment=True)
    val_set = common.FaceDataset(data["x_val"], data["y_val"])
    images, labels = common.scoring_data(data, args)
    scoring_sets = {name: common.FaceDataset(images[name], labels[name]) for name in images}

    model = CNN()
    with common.Timer() as train_timer:
        common.train_model(model, train_set, val_set, epochs=args.epochs, lr=LEARNING_RATE)
    common.warm_up(model, scoring_sets["test"])
    with common.Timer() as test_timer:
        common.predict(model, scoring_sets["test"])
    answers = {name: common.predict(model, scoring_sets[name]) for name in scoring_sets}

    common.report("cnn", model, train_timer.seconds, test_timer.seconds, labels, answers)
    torch.save(model.state_dict(), common.RESULTS_DIR / "cnn.pt")


if __name__ == "__main__":
    main()
