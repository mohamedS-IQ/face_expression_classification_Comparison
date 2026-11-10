import torch
import torch.nn as nn

import common

EPOCHS = 80
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
    internet_x, internet_y = common.load_internet_images()

    model = MLP()
    with common.Timer() as train_timer:
        common.train_model(model, train_set, val_set, epochs=args.epochs, lr=LEARNING_RATE)
    common.warm_up(model, test_set)
    with common.Timer() as test_timer:
        predictions = common.predict(model, test_set)
    internet = common.predict(model, common.FaceDataset(internet_x, internet_y))

    y_test = data["y_test"][:args.n_test]
    common.report("mlp", y_test, predictions, model, train_timer.seconds, test_timer.seconds, internet)
    torch.save(model.state_dict(), common.RESULTS_DIR / "mlp.pt")


if __name__ == "__main__":
    main()
