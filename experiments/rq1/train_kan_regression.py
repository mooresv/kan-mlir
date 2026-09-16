import torch
from kanlib.nn.bspline import Linear

torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)

DEVICE = torch.device("cuda")

N_TRAIN = 10000
N_CALIB = 2500
N_TEST = 5000

EPOCHS = 100
LR = 1.0e-3


def target_function(x):
    return (
        torch.sin(torch.pi * x[:, 0] * x[:, 1])
        + (x[:, 2] - 0.5) ** 2
        + 0.5 * x[:, 3]
    ).unsqueeze(1)


def make_data(n):
    x = 2.0 * torch.rand(n, 4) - 1.0
    y = target_function(x)
    return x, y


x_train, y_train = make_data(N_TRAIN)
x_calib, y_calib = make_data(N_CALIB)
x_test, y_test = make_data(N_TEST)

model = torch.nn.Sequential(
    Linear(4, 16, grid_size=5, spline_order=3),
    Linear(16, 16, grid_size=5, spline_order=3),
    Linear(16, 1, grid_size=5, spline_order=3),
).to(DEVICE)

x_train_gpu = x_train.to(DEVICE)
y_train_gpu = y_train.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training...")

train_dataset = torch.utils.data.TensorDataset(
    x_train,
    y_train,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
)

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    total_count = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.shape[0]
        total_count += xb.shape[0]

    train_mse = total_loss / total_count

    if epoch == 0 or (epoch + 1) % 10 == 0:
        model = model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            pred_calib = model(
                x_calib.to(DEVICE)
            )

            calib_mse = torch.mean(
                (
                    pred_calib
                    - y_calib.to(DEVICE)
                ) ** 2
            )

        print(
            f"epoch {epoch + 1:3d}: "
            f"train MSE={train_mse:.9e}  "
            f"calib MSE={calib_mse.item():.9e}"
        )

# KANLib operations during training can affect device placement.
model = model.to(DEVICE)
model.eval()

with torch.no_grad():
    pred_test = model(x_test.to(DEVICE))

    test_mse = torch.mean(
        (pred_test - y_test.to(DEVICE)) ** 2
    )

    test_rmse = torch.sqrt(test_mse)

print("\nOriginal model:")
print(f"  test MSE  = {test_mse.item():.12e}")
print(f"  test RMSE = {test_rmse.item():.12e}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "x_train": x_train,
        "y_train": y_train,
        "x_calib": x_calib,
        "y_calib": y_calib,
        "x_test": x_test,
        "y_test": y_test,
    },
    "trained_kan_regression.pt",
)

print("\nSaved trained_kan_regression.pt")
