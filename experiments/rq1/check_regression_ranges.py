import torch

from kanlib.nn.bspline import Linear


DEVICE = torch.device("cuda")

checkpoint = torch.load(
    "trained_kan_regression.pt",
    map_location="cpu",
)

x_test = checkpoint["x_test"]

model = torch.nn.Sequential(
    Linear(4, 16, grid_size=5, spline_order=3),
    Linear(16, 16, grid_size=5, spline_order=3),
    Linear(16, 1, grid_size=5, spline_order=3),
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

x = x_test.to(DEVICE)

print("Input to layer 0:")
print(f"  min = {x.min().item(): .9f}")
print(f"  max = {x.max().item(): .9f}")

with torch.no_grad():
    for i, layer in enumerate(model):
        # Per-feature input ranges.
        mins = x.min(dim=0).values
        maxs = x.max(dim=0).values

        outside = ((x < -1.0) | (x > 1.0)).float().mean()

        print(f"\nLayer {i} input:")
        print(f"  shape              = {tuple(x.shape)}")
        print(f"  global min         = {x.min().item(): .9f}")
        print(f"  global max         = {x.max().item(): .9f}")
        print(f"  fraction outside   = {outside.item():.9e}")

        print("  per-feature ranges:")
        for j in range(x.shape[1]):
            print(
                f"    {j:3d}: "
                f"[{mins[j].item(): .9f}, "
                f"{maxs[j].item(): .9f}]"
            )

        x = layer(x)

        print(f"\nLayer {i} output:")
        print(f"  global min         = {x.min().item(): .9f}")
        print(f"  global max         = {x.max().item(): .9f}")

print("\nFinal output:")
print(f"  min = {x.min().item(): .9f}")
print(f"  max = {x.max().item(): .9f}")
