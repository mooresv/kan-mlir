import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from kanlib.nn.bspline import Linear


SEED = 12345

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda")

BATCH_SIZE = 64
EPOCHS = 200
LR = 1.0e-3

GRID_SIZE = 5
SPLINE_ORDER = 3


# ============================================================
# Model
# ============================================================

def make_model():
    return torch.nn.Sequential(
        Linear(
            30,
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
# Load WDBC
# ============================================================

dataset = load_breast_cancer()

x = dataset.data.astype(np.float32)
y = dataset.target.astype(np.int64)

print("Wisconsin Diagnostic Breast Cancer")
print(f"examples: {x.shape[0]}")
print(f"features: {x.shape[1]}")
print(
    f"class counts: "
    f"0={np.sum(y == 0)} "
    f"1={np.sum(y == 1)}"
)


# ============================================================
# Stratified 60/20/20 split
#
# First reserve 40% for calibration + test.
# Then divide that equally.
# ============================================================

x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.40,
    random_state=SEED,
    stratify=y,
)

x_calib, x_test, y_calib, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)


# ============================================================
# Standardize using TRAINING statistics only
# ============================================================

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

# Defensive guard against constant features.
std[std == 0.0] = 1.0

x_train = (x_train - mean) / std
x_calib = (x_calib - mean) / std
x_test = (x_test - mean) / std


# ============================================================
# Convert to tensors
# ============================================================

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

x_calib = torch.tensor(x_calib, dtype=torch.float32)
y_calib = torch.tensor(y_calib, dtype=torch.long)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


print("\nSplit sizes:")
print(f"  training:    {len(x_train)}")
print(f"  calibration: {len(x_calib)}")
print(f"  test:        {len(x_test)}")

print("\nClass counts:")

for name, labels in [
    ("training", y_train),
    ("calibration", y_calib),
    ("test", y_test),
]:
    print(
        f"  {name:<11}: "
        f"class0={(labels == 0).sum().item():3d}  "
        f"class1={(labels == 1).sum().item():3d}"
    )


# ============================================================
# Training loader
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
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

print("\nTraining...")

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

print(
    f"  correct            = "
    f"{(test_pred == y_test.to(DEVICE)).sum().item()}"
    f"/{len(y_test)}"
)


# ============================================================
# Save everything needed for transformation experiment
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

        "feature_mean":
            torch.tensor(mean),

        "feature_std":
            torch.tensor(std),

        "feature_names":
            list(dataset.feature_names),

        "target_names":
            list(dataset.target_names),
    },
    "trained_kan_wdbc.pt",
)

print("\nSaved trained_kan_wdbc.pt")
