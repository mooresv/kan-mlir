import argparse
import torch

from kanlib.nn.bspline import Linear

def time_cuda(fn, warmup=100, iters=1000, profile_one=False):

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    # Profile exactly one warmed-up invocation.
    if profile_one:
        print("Profiling one warmed-up KANLib forward invocation")

        torch.cuda.nvtx.range_push("KANLIB_ONE_FORWARD")

        fn()

        # Ensure all kernels belonging to fn() finish before
        # closing the NVTX range.
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_pop()

        return None

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        fn()

    stop.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(stop)

    return total_ms / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="knot_poly_fits.pt",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 16384, 65536],
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--profile-one",
        action="store_true",
        help="After warmup, execute one forward inside an NVTX range and exit.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    data = torch.load(args.artifact, map_location="cpu")

    cfg = data["config"]
    grid = data["grid"].float()
    source_coeff = data["source_coefficients"].float()

    layer = Linear(
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        grid_size=cfg["grid_size"],
        spline_order=cfg["spline_order"],
        use_output_bias=False,
        use_residual_branch=False,
        use_spline_weight=False,
    ).to(device)

    with torch.no_grad():
        layer.coefficients.copy_(source_coeff.to(device))
        layer.basis.grid.copy_(grid.to(device))

    layer.eval()

    print("GPU:", torch.cuda.get_device_name())
    print("Din:", cfg["in_features"])
    print("Dout:", cfg["out_features"])
    print()
    print(f"{'batch':>10} {'time_us':>12} {'outputs/s':>15}")

    for batch in args.batches:
        torch.manual_seed(0)

        # Stay inside the fitted spline domain [-1,1].
        x = 2.0 * torch.rand(
            batch,
            cfg["in_features"],
            device=device,
            dtype=torch.float32,
        ) - 1.0

        with torch.no_grad():
            y = layer(x)

        def run():
            with torch.no_grad():
                layer(x)

        ms = time_cuda(
            run,
            warmup=args.warmup,
            iters=args.iters,
            profile_one=args.profile_one,
        )

        if args.profile_one:
            return

        us = 1000.0 * ms
        outputs_per_s = batch * cfg["out_features"] / (ms * 1.0e-3)

        print(
            f"{batch:10d} "
            f"{us:12.3f} "
            f"{outputs_per_s:15.3e}"
        )

        # Prevent accidental dead-code assumptions and make sure result is real.
        assert torch.isfinite(y).all()


if __name__ == "__main__":
    main()
