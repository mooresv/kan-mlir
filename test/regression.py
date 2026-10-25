import torch
from torch.utils.data import TensorDataset

from kanlib.nn.bspline import Linear
from kanlib.training import train

torch.manual_seed(1234)
device = torch.device("cuda")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def target(x):
    return (
        torch.sin(torch.pi * x[:, 0] * x[:, 1])
        + (x[:, 2] - 0.5) ** 2
        + 0.5 * x[:, 3]
    ).unsqueeze(1)

n_train = 10000
n_test = 5000

x_train = 2.0 * torch.rand(n_train, 4) - 1.0
x_test  = 2.0 * torch.rand(n_test, 4) - 1.0

y_train = target(x_train)
y_test  = target(x_test)

train_ds = TensorDataset(x_train, y_train)
test_ds  = TensorDataset(x_test, y_test)

# ------------------------------------------------------------
# KAN
# ------------------------------------------------------------

model = torch.nn.Sequential(
    Linear(
        in_features=4,
        out_features=16,
        grid_size=5,
        spline_order=3,
    ),
    Linear(
        in_features=16,
        out_features=16,
        grid_size=5,
        spline_order=3,
    ),
    Linear(
        in_features=16,
        out_features=1,
        grid_size=5,
        spline_order=3,
    ),
).to(device)

history = train(
    model=model,
    ds_train=train_ds,
    ds_val=test_ds,
    epochs=100,
    batch_size=256,
    loss_fn=torch.nn.MSELoss(),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
)

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

model = model.to(device)

model.eval()

print("\nTrained model structure:")
print(model)

for i, layer in enumerate(model):
    print(f"\n===== LAYER {i} =====")

    print("Parameters:")
    for name, param in layer.named_parameters():
        print(
            f"  {name:20s} "
            f"shape={tuple(param.shape)} "
            f"dtype={param.dtype} "
            f"device={param.device}"
        )

    print("Buffers:")
    for name, buf in layer.named_buffers():
        print(
            f"  {name:20s} "
            f"shape={tuple(buf.shape)} "
            f"dtype={buf.dtype} "
            f"device={buf.device}"
        )

        if name == "basis.grid":
            print("  grid:")
            print(buf)

with torch.no_grad():
    pred = model(x_test.to(device))
    mse = torch.mean((pred - y_test.to(device)) ** 2)
    rmse = torch.sqrt(mse)

print("test MSE :", mse.item())
print("test RMSE:", rmse.item())

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "x_test": x_test,
        "y_test": y_test,
    },
    "trained_kan_regression.pt",
)
