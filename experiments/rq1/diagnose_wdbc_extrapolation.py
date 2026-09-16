import torch
import torch.nn.functional as F

from kanlib.nn.bspline import Linear


DEVICE = torch.device("cuda")
CHECKPOINT = "trained_kan_wdbc.pt"

GRID_SIZE = 5
SPLINE_ORDER = 3

APPROX_PIECES = [1, 2, 4, 8]
FIT_SAMPLES_PER_PIECE = 64


def make_model():
    return torch.nn.Sequential(
        Linear(30, 16, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
        Linear(16, 16, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
        Linear(16, 2, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
    )


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

    return (
        Bi
        @ weighted_coeff[:, input_feature, :].T
    )


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

    C = torch.linalg.lstsq(A, values).solution

    return C.T.contiguous()


@torch.no_grad()
def convert_layer_approx(
    layer,
    calib_mins,
    calib_maxs,
    num_pieces,
):
    layer = layer.cpu()

    dout = layer.out_features
    din = layer.in_features

    grid = layer.basis.grid.detach().double()

    support_min = grid[:, 0].clone()
    support_max = grid[:, -1].clone()

    approx_min = torch.maximum(
        calib_mins,
        support_min,
    )

    approx_max = torch.minimum(
        calib_maxs,
        support_max,
    )

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
        xmin = approx_min[i].item()
        xmax = approx_max[i].item()

        if xmax <= xmin:
            raise RuntimeError(
                f"feature {i}: no intersection between "
                f"calibration range and spline support"
            )

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

            xs = left + (right - left) * fractions

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
        "boundaries": boundaries,
        "coeffs": coeffs,
        "support_min": support_min,
        "support_max": support_max,
        "weight_residual": (
            None
            if layer.weight_residual is None
            else layer.weight_residual
            .detach()
            .double()
            .clone()
        ),
    }


# ------------------------------------------------------------
# Evaluate ONE transformed spline edge.
#
# Returns the contribution from input feature i to output
# feature o, excluding the residual branch.
# ------------------------------------------------------------

def transformed_edge_forward(
    xi,
    converted,
    input_feature,
    output_feature,
):
    boundaries = converted["boundaries"].to(
        device=xi.device,
        dtype=xi.dtype,
    )

    coeffs = converted["coeffs"].to(
        device=xi.device,
        dtype=xi.dtype,
    )

    support_min = converted["support_min"].to(
        device=xi.device,
        dtype=xi.dtype,
    )

    support_max = converted["support_max"].to(
        device=xi.device,
        dtype=xi.dtype,
    )

    b = boundaries[input_feature]

    p = torch.searchsorted(
        b.contiguous(),
        xi.contiguous(),
        right=True,
    ) - 1

    p = p.clamp(
        0,
        coeffs.shape[2] - 1,
    )

    left = b[p]
    t = xi - left

    c = coeffs[
        output_feature,
        input_feature,
        p,
        :
    ]

    value = (
        (
            c[:, 3] * t
            + c[:, 2]
        ) * t
        + c[:, 1]
    ) * t + c[:, 0]

    valid = (
        (xi >= support_min[input_feature])
        & (xi < support_max[input_feature])
    )

    return value * valid


# ------------------------------------------------------------
# Original learned spline contribution for ONE edge.
# ------------------------------------------------------------

@torch.no_grad()
def original_edge_forward(
    layer,
    x,
    input_feature,
    output_feature,
):
    layer = layer.cpu()

    xi = x[:, input_feature].detach().cpu()

    din = layer.in_features

    sample = torch.zeros(
        x.shape[0],
        din,
        dtype=xi.dtype,
    )

    sample[:, input_feature] = xi

    B = layer.basis(sample)

    Bi = B[:, input_feature, :]

    coeff = (
        layer.weighted_coefficients[
            output_feature,
            input_feature,
            :
        ]
        .detach()
        .cpu()
    )

    return Bi @ coeff


# ------------------------------------------------------------
# Complete transformed layer.
# ------------------------------------------------------------

def polynomial_layer_forward(x, converted):
    boundaries = converted["boundaries"].to(
        device=x.device,
        dtype=x.dtype,
    )

    coeffs = converted["coeffs"].to(
        device=x.device,
        dtype=x.dtype,
    )

    support_min = converted["support_min"].to(
        device=x.device,
        dtype=x.dtype,
    )

    support_max = converted["support_max"].to(
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

    output = torch.zeros(
        batch,
        dout,
        device=x.device,
        dtype=x.dtype,
    )

    for i in range(din):
        xi = x[:, i]
        b = boundaries[i]

        p = torch.searchsorted(
            b.contiguous(),
            xi.contiguous(),
            right=True,
        ) - 1

        p = p.clamp(
            0,
            num_pieces - 1,
        )

        left = b[p]
        t = xi - left

        c = coeffs[
            :,
            i,
            p,
            :
        ].permute(1, 0, 2)

        value = (
            (
                c[:, :, 3] * t[:, None]
                + c[:, :, 2]
            ) * t[:, None]
            + c[:, :, 1]
        ) * t[:, None] + c[:, :, 0]

        valid = (
            (xi >= support_min[i])
            & (xi < support_max[i])
        )

        output += value * valid[:, None]

    if weight_residual is not None:
        output += F.linear(
            F.silu(x),
            weight_residual,
        )

    return output


# ------------------------------------------------------------
# Diagnose maximum local error.
# ------------------------------------------------------------

@torch.no_grad()
def diagnose_layer(
    layer_index,
    layer,
    x,
    converted,
):
    original_output = layer(x)

    transformed_output = polynomial_layer_forward(
        x,
        converted,
    )

    error = (
        transformed_output
        - original_output
    )

    abs_error = error.abs()

    flat_index = abs_error.argmax().item()

    batch_index = (
        flat_index // layer.out_features
    )

    output_feature = (
        flat_index % layer.out_features
    )

    max_error = error[
        batch_index,
        output_feature,
    ].item()

    #
    # Now decompose this output error by input edge.
    #
    edge_info = []

    for i in range(layer.in_features):
        original_edge = original_edge_forward(
            layer,
            x,
            i,
            output_feature,
        )

        transformed_edge = transformed_edge_forward(
            x[:, i],
            converted,
            i,
            output_feature,
        ).cpu()

        edge_error = (
            transformed_edge[batch_index]
            - original_edge[batch_index]
        ).item()

        edge_info.append(
            (abs(edge_error), edge_error, i)
        )

    edge_info.sort(reverse=True)

    _, largest_edge_error, input_feature = (
        edge_info[0]
    )

    xi = x[
        batch_index,
        input_feature,
    ].item()

    b = converted["boundaries"][
        input_feature
    ]

    approx_min = b[0].item()
    approx_max = b[-1].item()

    support_min = converted[
        "support_min"
    ][input_feature].item()

    support_max = converted[
        "support_max"
    ][input_feature].item()

    inside_approx = (
        approx_min <= xi <= approx_max
    )

    inside_support = (
        support_min <= xi < support_max
    )

    print(
        f"layer {layer_index}:"
    )

    print(
        f"  maximum local output error = "
        f"{max_error:+.9e}"
    )

    print(
        f"  |maximum local error|      = "
        f"{abs(max_error):.9e}"
    )

    print(
        f"  test example              = "
        f"{batch_index}"
    )

    print(
        f"  output feature            = "
        f"{output_feature}"
    )

    print(
        f"  largest edge contributor  = "
        f"input feature {input_feature}"
    )

    print(
        f"  edge error                = "
        f"{largest_edge_error:+.9e}"
    )

    print(
        f"  activation                = "
        f"{xi:+.9e}"
    )

    print(
        f"  approximation interval    = "
        f"[{approx_min:+.9e}, "
        f"{approx_max:+.9e}]"
    )

    print(
        f"  spline support            = "
        f"[{support_min:+.9e}, "
        f"{support_max:+.9e}]"
    )

    print(
        f"  inside approximation?     = "
        f"{inside_approx}"
    )

    print(
        f"  inside spline support?    = "
        f"{inside_support}"
    )

    #
    # Also print top five edge contributors. This guards
    # against incorrectly attributing an output error to one
    # edge when several moderate edge errors add together.
    #
    print(
        "  top edge-error contributors:"
    )

    for rank, (_, signed_error, i) in enumerate(
        edge_info[:5],
        start=1,
    ):
        value = x[
            batch_index,
            i,
        ].item()

        bi = converted["boundaries"][i]

        amin = bi[0].item()
        amax = bi[-1].item()

        smin = converted[
            "support_min"
        ][i].item()

        smax = converted[
            "support_max"
        ][i].item()

        in_approx = amin <= value <= amax
        in_support = smin <= value < smax

        print(
            f"    {rank}: "
            f"feature={i:2d}  "
            f"error={signed_error:+.9e}  "
            f"x={value:+.9e}  "
            f"in_approx={str(in_approx):5s}  "
            f"in_support={str(in_support):5s}"
        )


# ============================================================
# Main
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

x_calib = checkpoint["x_calib"].to(DEVICE)
x_test = checkpoint["x_test"].to(DEVICE)

model = make_model()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print("Collecting calibration activation ranges...")

calib_inputs = collect_layer_inputs(
    model,
    x_calib,
)

ranges = compute_ranges(
    calib_inputs,
)

test_inputs = collect_layer_inputs(
    model,
    x_test,
)


for pieces in APPROX_PIECES:
    print()
    print("=" * 78)
    print(
        f"{pieces}-PIECE REPRESENTATION"
    )
    print("=" * 78)

    converted_layers = []

    for layer_index, layer in enumerate(model):
        mins, maxs = ranges[layer_index]

        converted = convert_layer_approx(
            layer,
            mins,
            maxs,
            pieces,
        )

        converted_layers.append(converted)

        model = model.to(DEVICE)

    for layer_index, layer in enumerate(model):
        print()

        diagnose_layer(
            layer_index,
            layer,
            test_inputs[layer_index],
            converted_layers[layer_index],
        )

        model = model.to(DEVICE)
