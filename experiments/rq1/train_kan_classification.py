import torch
from torch.utils.data import TensorDataset, DataLoader

from kanlib.nn.bspline import Linear


torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)

DEVICE = torch.device("cuda")

N_TRAIN = 10000
N_CALIB = 2500
N_TEST = 5000

BATCH_SIZE = 256
EPOCHS = 100
LR = 1.0e-3

GRID_SIZE = 5
SPLINE_ORDER = 3


# ============================================================
# Classification problem
#
# Two concentric classes:
#
#     class 0: inner disk
#     class 1: outer region
#
# Inputs are uniform on [-1,1]^2.
#
# Boundary:
#
#     x1^2 + x2^2 = RADIUS^2
# ============================================================

RADIUS = 0.70


def make_data(n):
    x = 2.0 * torch.rand(n, 2) - 1.0

    r2 = x[:, 0] ** 2 + x[:, 1] ** 2

    y = (r2 >= RADIUS ** 2).long()

    return x, y


def make_model():
    return torch.nn.Sequential(
        Linear(
            2,
            16,
            grid_size=GRID_SIZE,
            spline_order=SPLINE_ORDER,
        ),
        Linear(
            16,
            16,
            grid_size=GRID_SIZE,
            spline_order=SPLINE_ORDER,
        ),
        Linear(
            16,
            2,
            grid_size=GRID_SIZE,
            spline_order=SPLINE_ORDER,
        ),
    )


# ============================================================
# Data
# ============================================================

x_train, y_train = make_data(N_TRAIN)
x_calib, y_calib = make_data(N_CALIB)
x_test, y_test = make_data(N_TEST)

train_dataset = TensorDataset(
    x_train,
    y_train,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ============================================================
# Model
# ============================================================

model = make_model().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)


# ============================================================
# Training
# ============================================================

print("Training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_count = 0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)

        loss = torch.nn.functional.cross_entropy(
            logits,
            yb,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * xb.shape[0]
        )

        total_count += xb.shape[0]

    train_loss = total_loss / total_count

    if epoch == 0 or (epoch + 1) % 10 == 0:

        model = model.to(DEVICE)
        model.eval()

        with torch.no_grad():

            calib_logits = model(
                x_calib.to(DEVICE)
            )

            calib_loss = (
                torch.nn.functional.cross_entropy(
                    calib_logits,
                    y_calib.to(DEVICE),
                )
            )

            calib_pred = (
                calib_logits.argmax(dim=1)
            )

            calib_accuracy = (
                calib_pred
                == y_calib.to(DEVICE)
            ).float().mean()

        print(
            f"epoch {epoch + 1:3d}: "
            f"train loss={train_loss:.9e}  "
            f"calib loss={calib_loss.item():.9e}  "
            f"calib accuracy="
            f"{100.0 * calib_accuracy.item():.4f}%"
        )


# ============================================================
# Final independent test evaluation
# ============================================================

model = model.to(DEVICE)
model.eval()

with torch.no_grad():

    test_logits = model(
        x_test.to(DEVICE)
    )

    test_loss = (
        torch.nn.functional.cross_entropy(
            test_logits,
            y_test.to(DEVICE),
        )
    )

    test_pred = test_logits.argmax(dim=1)

    test_accuracy = (
        test_pred
        == y_test.to(DEVICE)
    ).float().mean()


print("\nOriginal model:")

print(
    f"  test cross entropy = "
    f"{test_loss.item():.12e}"
)

print(
    f"  test accuracy      = "
    f"{100.0 * test_accuracy.item():.6f}%"
)


# ============================================================
# Class balance
# ============================================================

for split_name, labels in [
    ("train", y_train),
    ("calibration", y_calib),
    ("test", y_test),
]:

    count0 = (labels == 0).sum().item()
    count1 = (labels == 1).sum().item()

    print(
        f"  {split_name:11s}: "
        f"class0={count0:5d}  "
        f"class1={count1:5d}"
    )


# ============================================================
# Save
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "x_train": x_train,
        "y_train": y_train,

        "x_calib": x_calib,
        "y_calib": y_calib,

        "x_test": x_test,
        "y_test": y_test,

        "radius": RADIUS,
    },
    "trained_kan_classification.pt",
)

print(
    "\nSaved trained_kan_classification.pt"
)
