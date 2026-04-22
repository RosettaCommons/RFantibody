"""Diff two RF2 checkpoints on state-dict keys and tensor shapes.

Usage:
    python scripts/diff_rf2_checkpoints.py <ckpt_a> <ckpt_b>
"""
import sys
from pathlib import Path

import torch


def load_state_dict(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a = load_state_dict(a_path)
    b = load_state_dict(b_path)

    a_keys, b_keys = set(a.keys()), set(b.keys())
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    shared = sorted(a_keys & b_keys)
    shape_mismatch = [
        (k, tuple(a[k].shape), tuple(b[k].shape))
        for k in shared
        if hasattr(a[k], "shape") and hasattr(b[k], "shape") and a[k].shape != b[k].shape
    ]

    print(f"=== {a_path.name}  vs  {b_path.name} ===")
    print(f"A total keys: {len(a_keys)}")
    print(f"B total keys: {len(b_keys)}")
    print(f"Shared keys : {len(shared)}")
    print(f"Only in A   : {len(only_a)}")
    print(f"Only in B   : {len(only_b)}")
    print(f"Shape mismatch on shared: {len(shape_mismatch)}")

    def _preview(label, items, limit=25):
        print(f"\n--- {label} (showing up to {limit}) ---")
        for x in items[:limit]:
            print(x)
        if len(items) > limit:
            print(f"... {len(items) - limit} more")

    _preview("Only in A", only_a)
    _preview("Only in B", only_b)
    _preview("Shape mismatch (key, A_shape, B_shape)", shape_mismatch)

    verdict = "DROP-IN COMPATIBLE" if not only_a and not only_b and not shape_mismatch else "NOT DROP-IN COMPATIBLE"
    print(f"\nVerdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
