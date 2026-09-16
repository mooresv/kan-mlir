import torch

from kanlib.nn.bspline import Linear


DEVICE = torch.device("cuda")
CHECKPOINT = "trained_kan_wdbc.pt"

GRID_SIZE = 5
SPLINE_ORDER = 3


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


@torch.no_grad()
def collect_layer_inputs(model, x):
    inputs = []
    current = x

    for layer in model:
        inputs.append(current.detach().clone())
        current = layer(current)

    return inputs


def analyze_split(name, layer_inputs, model):
    print(f"\n{name}")
    print("=" * 78)

    for layer_index, (x, layer) in enumerate(
        zip(layer_inputs, model)
    ):
        grid = layer.basis.grid.to(
            device=x.device,
            dtype=x.dtype,
        )

        #
        # Full support of each input feature's retained
        # B-spline basis functions.
        #
        support_min = grid[:, 0]
        support_max = grid[:, -1]

        below = x < support_min
        above = x > support_max
        outside = below | above

        count = outside.sum().item()
        total = x.numel()

        examples_any_outside = (
            outside.any(dim=1).sum().item()
        )

        #
        # Distance beyond spline support.
        #
        below_excess = torch.where(
            below,
            support_min - x,
            torch.zeros_like(x),
        )

        above_excess = torch.where(
            above,
            x - support_max,
            torch.zeros_like(x),
        )

        excess = torch.maximum(
            below_excess,
            above_excess,
        )

        max_excess = excess.max().item()

        #
        # Actual activation range.
        #
        global_min = x.min().item()
        global_max = x.max().item()

        print(
            f"layer {layer_index}: "
            f"{layer.in_features} -> "
            f"{layer.out_features}"
        )

        print(
            f"  activation range: "
            f"[{global_min:.6f}, "
            f"{global_max:.6f}]"
        )

        print(
            f"  spline support:    "
            f"[{support_min.min().item():.6f}, "
            f"{support_max.max().item():.6f}]"
        )

        print(
            f"  outside support:   "
            f"{count}/{total} "
            f"({100.0 * count / total:.6f}%)"
        )

        print(
            f"  examples with >=1 feature outside: "
            f"{examples_any_outside}/{x.shape[0]} "
            f"({100.0 * examples_any_outside / x.shape[0]:.6f}%)"
        )

        print(
            f"  maximum excess:    "
            f"{max_excess:.9e}"
        )

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = make_model()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


for split_name in [
    "x_train",
    "x_calib",
    "x_test",
]:
    x = checkpoint[split_name].to(DEVICE)

    layer_inputs = collect_layer_inputs(
        model,
        x,
    )

    analyze_split(
        split_name,
        layer_inputs,
        model,
    )
