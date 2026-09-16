import torch
import torch.nn.functional as F

from kanlib.nn.bspline import Linear


DEVICE = torch.device("cuda")

CHECKPOINT = "trained_kan_classification.pt"

GRID_SIZE = 5
SPLINE_ORDER = 3

APPROX_PIECES = [1, 2, 4, 8]
FIT_SAMPLES_PER_PIECE = 64


# ============================================================
# Model
# ============================================================

def make_model():
    return torch.nn.Sequential(
        Linear(2, 16, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
        Linear(16, 16, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
        Linear(16, 2, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
    )


# ============================================================
# Collect inputs to each layer from original model
# ============================================================

@torch.no_grad()
def collect_layer_inputs(model, x):
    inputs = []
    current = x

    for layer in model:
        inputs.append(current.detach().clone())
        current = layer(current)

    return inputs


def compute_ranges(layer_inputs):
    ranges = []

    for x in layer_inputs:
        mins = x.min(dim=0).values.detach().cpu().double()
        maxs = x.max(dim=0).values.detach().cpu().double()
        ranges.append((mins, maxs))

    return ranges


# ============================================================
# Independent test coverage of calibration ranges
# ============================================================

def analyze_range_coverage(test_inputs, ranges):
    coverage = []

    for layer_index, x in enumerate(test_inputs):
        mins, maxs = ranges[layer_index]

        mins_gpu = mins.to(
            device=x.device,
            dtype=x.dtype,
        )

        maxs_gpu = maxs.to(
            device=x.device,
            dtype=x.dtype,
        )

        below = x < mins_gpu
        above = x > maxs_gpu
        outside = below | above

        count = outside.sum().item()
        total = x.numel()

        below_distance = torch.where(
            below,
            mins_gpu - x,
            torch.zeros_like(x),
        )

        above_distance = torch.where(
            above,
            x - maxs_gpu,
            torch.zeros_like(x),
        )

        max_excess = torch.maximum(
            below_distance,
            above_distance,
        ).max().item()

        coverage.append(
            {
                "count": count,
                "total": total,
                "fraction": count / total,
                "max_excess": max_excess,
            }
        )

    return coverage


# ============================================================
# Evaluate learned spline edge functions
# ============================================================

@torch.no_grad()
def evaluate_spline_edges(layer, input_feature, xs):
    layer = layer.cpu()

    din = layer.in_features

    sample = torch.zeros(
        xs.numel(),
        din,
        dtype=torch.float64,
    )

    sample[:, input_feature] = xs

    original_grid = layer.basis.grid
    layer.basis.grid = original_grid.double()

    B = layer.basis(sample)

    layer.basis.grid = original_grid

    Bi = B[:, input_feature, :]

    weighted_coeff = (
        layer.weighted_coefficients
        .detach()
        .double()
    )

    values = (
        Bi
        @ weighted_coeff[:, input_feature, :].T
    )

    return values


# ============================================================
# Cubic least-squares fit
# ============================================================

def fit_cubic_least_squares(xs, values, left):
    t = xs - left

    A = torch.stack(
        [
            torch.ones_like(t),
            t,
            t * t,
            t * t * t,
        ],
        dim=1,
    )

    C = torch.linalg.lstsq(
        A,
        values,
    ).solution

    return C.T.contiguous()


# ============================================================
# Approximate representation
# ============================================================

@torch.no_grad()
def convert_layer_approx(
    layer,
    mins,
    maxs,
    num_pieces,
):
    layer = layer.cpu()

    dout = layer.out_features
    din = layer.in_features

    boundaries = torch.empty(
        din,
        num_pieces + 1,
        dtype=torch.float64,
    )

    coeffs = torch.empty(
        dout,
        din,
        num_pieces,
        4,
        dtype=torch.float64,
    )

    for i in range(din):
        xmin = mins[i].item()
        xmax = maxs[i].item()

        if xmax <= xmin:
            xmax = xmin + 1.0e-8

        feature_boundaries = torch.linspace(
            xmin,
            xmax,
            num_pieces + 1,
            dtype=torch.float64,
        )

        boundaries[i] = feature_boundaries

        for p in range(num_pieces):
            left = feature_boundaries[p].item()
            right = feature_boundaries[p + 1].item()

            fractions = (
                torch.arange(
                    FIT_SAMPLES_PER_PIECE,
                    dtype=torch.float64,
                )
                + 0.5
            ) / FIT_SAMPLES_PER_PIECE

            xs = (
                left
                + (right - left) * fractions
            )

            values = evaluate_spline_edges(
                layer,
                i,
                xs,
            )

            coeffs[:, i, p, :] = (
                fit_cubic_least_squares(
                    xs,
                    values,
                    left,
                )
            )

    return {
        "kind": f"{num_pieces}-piece",
        "boundaries": boundaries,
        "coeffs": coeffs,
        "weight_residual": (
            None
            if layer.weight_residual is None
            else layer.weight_residual
            .detach()
            .double()
            .clone()
        ),
    }


# ============================================================
# Exact knot-aligned representation
# ============================================================

@torch.no_grad()
def convert_layer_exact(layer):
    layer = layer.cpu()

    dout = layer.out_features
    din = layer.in_features

    grid = layer.basis.grid.detach().double()

    num_pieces = grid.shape[1] - 1

    coeffs = torch.empty(
        dout,
        din,
        num_pieces,
        4,
        dtype=torch.float64,
    )

    fractions = torch.tensor(
        [0.125, 0.375, 0.625, 0.875],
        dtype=torch.float64,
    )

    for i in range(din):
        for p in range(num_pieces):
            left = grid[i, p].item()
            right = grid[i, p + 1].item()

            xs = (
                left
                + (right - left) * fractions
            )

            values = evaluate_spline_edges(
                layer,
                i,
                xs,
            )

            t = xs - left

            A = torch.stack(
                [
                    torch.ones_like(t),
                    t,
                    t * t,
                    t * t * t,
                ],
                dim=1,
            )

            C = torch.linalg.solve(
                A,
                values,
            )

            coeffs[:, i, p, :] = C.T

    return {
        "kind": "exact-knot",
        "boundaries": grid,
        "coeffs": coeffs,
        "weight_residual": (
            None
            if layer.weight_residual is None
            else layer.weight_residual
            .detach()
            .double()
            .clone()
        ),
    }


# ============================================================
# Execute transformed layer
# ============================================================

def polynomial_layer_forward(x, converted):
    boundaries = converted["boundaries"].to(
        device=x.device,
        dtype=x.dtype,
    )

    coeffs = converted["coeffs"].to(
        device=x.device,
        dtype=x.dtype,
    )

    weight_residual = converted["weight_residual"]

    if weight_residual is not None:
        weight_residual = weight_residual.to(
            device=x.device,
            dtype=x.dtype,
        )

    batch, din = x.shape
    dout = coeffs.shape[0]
    num_pieces = coeffs.shape[2]

    spline_output = torch.zeros(
        batch,
        dout,
        device=x.device,
        dtype=x.dtype,
    )

    exact = converted["kind"] == "exact-knot"

    for i in range(din):
        xi = x[:, i]
        b = boundaries[i]

        p = torch.searchsorted(
            b.contiguous(),
            xi.contiguous(),
            right=True,
        ) - 1

        p_safe = p.clamp(
            0,
            num_pieces - 1,
        )

        left = b[p_safe]
        t = xi - left

        c = coeffs[
            :,
            i,
            p_safe,
            :
        ].permute(1, 0, 2)

        value = (
            (
                c[:, :, 3] * t[:, None]
                + c[:, :, 2]
            ) * t[:, None]
            + c[:, :, 1]
        ) * t[:, None] + c[:, :, 0]

        if exact:
            valid = (
                (xi >= b[0])
                & (xi < b[-1])
            )

            value = value * valid[:, None]

        spline_output += value

    if weight_residual is not None:
        spline_output += F.linear(
            F.silu(x),
            weight_residual,
        )

    return spline_output


# ============================================================
# Classification evaluation
# ============================================================

@torch.no_grad()
def evaluate_representation(
    model,
    converted_layers,
    x_test,
    y_test,
):
    x_orig = x_test
    x_poly = x_test.clone()

    local_errors = []
    accumulated_errors = []

    for i, layer in enumerate(model):

        y_orig = layer(x_orig)

        #
        # Local error:
        # original and transformed layer get same input.
        #
        y_poly_local = polynomial_layer_forward(
            x_orig,
            converted_layers[i],
        )

        local_diff = y_poly_local - y_orig

        local_errors.append(
            (
                local_diff.abs().max().item(),
                torch.sqrt(
                    torch.mean(
                        local_diff * local_diff
                    )
                ).item(),
            )
        )

        #
        # Accumulated transformed-network error.
        #
        y_poly = polynomial_layer_forward(
            x_poly,
            converted_layers[i],
        )

        accum_diff = y_poly - y_orig

        accumulated_errors.append(
            (
                accum_diff.abs().max().item(),
                torch.sqrt(
                    torch.mean(
                        accum_diff * accum_diff
                    )
                ).item(),
            )
        )

        x_orig = y_orig
        x_poly = y_poly

    original_logits = x_orig
    transformed_logits = x_poly

    original_pred = original_logits.argmax(dim=1)
    transformed_pred = transformed_logits.argmax(dim=1)

    original_accuracy = (
        original_pred == y_test
    ).float().mean()

    transformed_accuracy = (
        transformed_pred == y_test
    ).float().mean()

    agreement = (
        transformed_pred == original_pred
    ).float().mean()

    changed = (
        transformed_pred != original_pred
    )

    changed_count = changed.sum().item()

    diff = (
        transformed_logits
        - original_logits
    )

    max_abs = diff.abs().max()

    rms = torch.sqrt(
        torch.mean(diff * diff)
    )

    original_rms = torch.sqrt(
        torch.mean(
            original_logits
            * original_logits
        )
    )

    relative_rms = rms / original_rms

    #
    # Binary classification margin:
    #
    #   |logit_1 - logit_0|
    #
    # A small margin means the original model is close
    # to its decision boundary.
    #
    original_margin = (
        original_logits[:, 1]
        - original_logits[:, 0]
    ).abs()

    if changed_count > 0:
        changed_margins = (
            original_margin[changed]
        )

        changed_margin_mean = (
            changed_margins.mean().item()
        )

        changed_margin_max = (
            changed_margins.max().item()
        )

        changed_margin_min = (
            changed_margins.min().item()
        )

    else:
        changed_margin_mean = 0.0
        changed_margin_max = 0.0
        changed_margin_min = 0.0

    return {
        "original_accuracy":
            original_accuracy.item(),

        "accuracy":
            transformed_accuracy.item(),

        "agreement":
            agreement.item(),

        "changed":
            changed_count,

        "max_abs":
            max_abs.item(),

        "rms":
            rms.item(),

        "relative_rms":
            relative_rms.item(),

        "local_errors":
            local_errors,

        "accumulated_errors":
            accumulated_errors,

        "changed_margin_mean":
            changed_margin_mean,

        "changed_margin_min":
            changed_margin_min,

        "changed_margin_max":
            changed_margin_max,
    }


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

x_calib = checkpoint["x_calib"].to(DEVICE)

x_test = checkpoint["x_test"].to(DEVICE)
y_test = checkpoint["y_test"].to(DEVICE)

model = make_model()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# Calibration ranges
# ============================================================

print(
    "Collecting CALIBRATION activation ranges..."
)

calib_inputs = collect_layer_inputs(
    model,
    x_calib,
)

ranges = compute_ranges(
    calib_inputs,
)

for i, (mins, maxs) in enumerate(ranges):

    print(
        f"  layer {i}: "
        f"global range "
        f"[{mins.min().item():.6f}, "
        f"{maxs.max().item():.6f}]"
    )


# ============================================================
# Test coverage
# ============================================================

print(
    "\nChecking TEST activations against "
    "calibration ranges..."
)

test_inputs = collect_layer_inputs(
    model,
    x_test,
)

coverage = analyze_range_coverage(
    test_inputs,
    ranges,
)

for i, c in enumerate(coverage):

    print(
        f"  layer {i}: "
        f"outside={c['count']}/{c['total']} "
        f"({100.0 * c['fraction']:.6f}%)  "
        f"max_excess={c['max_excess']:.9e}"
    )


# ============================================================
# Build approximate representations
# ============================================================

representations = {}

for pieces in APPROX_PIECES:

    print(
        f"\nFitting {pieces}-piece "
        f"representation using CALIBRATION ranges..."
    )

    converted_layers = []

    for i, layer in enumerate(model):

        mins, maxs = ranges[i]

        converted = convert_layer_approx(
            layer,
            mins,
            maxs,
            pieces,
        )

        converted_layers.append(
            converted
        )

        print(
            f"  layer {i}: "
            f"{layer.in_features} -> "
            f"{layer.out_features}"
        )

    representations[
        f"{pieces}-piece"
    ] = converted_layers

    model = model.to(DEVICE)


# ============================================================
# Exact representation
# ============================================================

print(
    "\nConstructing exact knot-aligned "
    "representation..."
)

exact_layers = []

for i, layer in enumerate(model):

    converted = convert_layer_exact(
        layer
    )

    exact_layers.append(
        converted
    )

    print(
        f"  layer {i}: "
        f"{layer.in_features} -> "
        f"{layer.out_features}, "
        f"{converted['coeffs'].shape[2]} pieces"
    )

representations[
    "exact-knot"
] = exact_layers

model = model.to(DEVICE)


# ============================================================
# Evaluate
# ============================================================

results = {}

print(
    "\nEvaluating all representations "
    "on TEST set..."
)

for name, converted_layers in (
    representations.items()
):

    print(f"  {name}")

    results[name] = (
        evaluate_representation(
            model,
            converted_layers,
            x_test,
            y_test,
        )
    )


# ============================================================
# Local errors
# ============================================================

print("\n")
print("=" * 78)
print(
    "LOCAL LAYER REPRESENTATION ERROR -- TEST SET"
)
print("=" * 78)

for name in representations:

    print(f"\n{name}")

    for i, (max_abs, rms) in enumerate(
        results[name]["local_errors"]
    ):

        print(
            f"  layer {i}: "
            f"max_abs={max_abs:.9e}  "
            f"rms={rms:.9e}"
        )


# ============================================================
# Accumulated errors
# ============================================================

print("\n")
print("=" * 78)
print(
    "ACCUMULATED NETWORK ERROR -- TEST SET"
)
print("=" * 78)

for name in representations:

    print(f"\n{name}")

    for i, (max_abs, rms) in enumerate(
        results[name][
            "accumulated_errors"
        ]
    ):

        print(
            f"  through layer {i}: "
            f"max_abs={max_abs:.9e}  "
            f"rms={rms:.9e}"
        )


# ============================================================
# Main classification table
# ============================================================

original_accuracy = next(
    iter(results.values())
)["original_accuracy"]

print("\n")
print("=" * 112)
print(
    "END-TO-END CLASSIFICATION COMPARISON"
)
print("=" * 112)

print(
    f"\nOriginal KAN test accuracy: "
    f"{100.0 * original_accuracy:.6f}%\n"
)

header = (
    f"{'Representation':<18}"
    f"{'Pieces':>8}"
    f"{'Accuracy':>14}"
    f"{'Agreement':>14}"
    f"{'Changed':>10}"
    f"{'RMS(logits)':>18}"
    f"{'Max |diff|':>18}"
    f"{'Relative RMS':>18}"
)

print(header)
print("-" * len(header))

for name in [
    "1-piece",
    "2-piece",
    "4-piece",
    "8-piece",
    "exact-knot",
]:

    r = results[name]

    pieces = (
        "11"
        if name == "exact-knot"
        else name.split("-")[0]
    )

    print(
        f"{name:<18}"
        f"{pieces:>8}"
        f"{100.0 * r['accuracy']:>13.6f}%"
        f"{100.0 * r['agreement']:>13.6f}%"
        f"{r['changed']:>10d}"
        f"{r['rms']:>18.9e}"
        f"{r['max_abs']:>18.9e}"
        f"{r['relative_rms']:>18.9e}"
    )


# ============================================================
# Accuracy change
# ============================================================

print(
    "\nAccuracy change relative to original:"
)

for name in [
    "1-piece",
    "2-piece",
    "4-piece",
    "8-piece",
    "exact-knot",
]:

    r = results[name]

    delta = (
        r["accuracy"]
        - original_accuracy
    )

    print(
        f"  {name:<12} "
        f"delta="
        f"{100.0 * delta:+.6f} "
        f"percentage points"
    )


# ============================================================
# Changed-prediction margins
# ============================================================

print(
    "\nOriginal-model margins for "
    "changed predictions:"
)

for name in [
    "1-piece",
    "2-piece",
    "4-piece",
    "8-piece",
    "exact-knot",
]:

    r = results[name]

    print(
        f"  {name:<12} "
        f"changed={r['changed']:4d}  "
        f"margin_min="
        f"{r['changed_margin_min']:.9e}  "
        f"margin_mean="
        f"{r['changed_margin_mean']:.9e}  "
        f"margin_max="
        f"{r['changed_margin_max']:.9e}"
    )
