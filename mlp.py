import torch.nn as nn

import common

EPOCHS = 30
LEARNING_RATE = 1e-3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(common.EMOTIONS)),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    args = common.parse_args(default_epochs=EPOCHS)
    common.set_seed()
    data = common.load_data()
    train_set = common.FaceDataset(data["x_train"][:args.n_train], data["y_train"][:args.n_train])
    val_set = common.FaceDataset(data["x_val"], data["y_val"])
    test_set = common.FaceDataset(data["x_test"][:args.n_test], data["y_test"][:args.n_test])

    model = MLP()
    with common.Timer() as train_timer:
        common.train_model(model, train_set, val_set, epochs=args.epochs, lr=LEARNING_RATE)
    with common.Timer() as test_timer:
        predictions = common.predict(model, test_set)

    common.report("mlp", data["y_test"][:args.n_test], predictions, model, train_timer.seconds, test_timer.seconds)


if __name__ == "__main__":
    main()
