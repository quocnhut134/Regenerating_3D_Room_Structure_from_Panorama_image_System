#!/usr/bin/env python3
"""Convert raw panorama boundary JSON to post-processed layout3d outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .dula.layout import fit_layout
except ImportError:  # Allows direct CLI execution: python postprocessing/...
    from Computer_Graphics.Project.src.postprocessing.dula.layout import fit_layout


def boundary_to_floor_xz(
    floor_boundary: list[float] | np.ndarray,
    image_size: list[int] | tuple[int, int],
    camera_height: float = 1.6,
    smooth_window: int = 9,
) -> np.ndarray:
    """Project equirectangular floor-boundary pixels onto Y=-camera_height."""
    height, width = image_size
    y = np.asarray(floor_boundary, dtype=np.float64)
    if len(y) != width:
        raise ValueError(f"floor_boundary length {len(y)} does not match image width {width}")

    if smooth_window > 1:
        y = circular_smooth(y, smooth_window)

    columns = np.arange(width, dtype=np.float64)
    theta = (columns / width - 0.5) * 2.0 * np.pi
    phi = (0.5 - y / height) * np.pi

    tan_phi = np.tan(phi)
    eps = 1e-6
    tan_phi = np.where(np.abs(tan_phi) < eps, np.sign(tan_phi) * eps, tan_phi)
    tan_phi = np.where(tan_phi == 0.0, eps, tan_phi)

    d = camera_height / -tan_phi
    return np.column_stack([d * np.sin(theta), d * np.cos(theta)])


def circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.r_[values[-pad:], values, values[:pad]]
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def remove_duplicate_neighbors(points: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if len(points) == 0:
        return points
    clean = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - clean[-1]) > eps:
            clean.append(point)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) <= eps:
        clean.pop()
    return np.asarray(clean, dtype=np.float64)


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    z = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    points = remove_duplicate_neighbors(points)
    if len(points) >= 3 and signed_area(points) < 0:
        return points[::-1]
    return points


def save_layout(
    corners_xz: np.ndarray,
    output_path: str | Path,
    camera_height: float = 1.6,
    layout_height: float = 2.8,
) -> None:
    corners_xz = ensure_ccw(np.asarray(corners_xz, dtype=np.float64))
    payload = {
        "camera_height": float(camera_height),
        "layout_height": float(layout_height),
        "coordinate_system": {
            "camera": [0.0, 0.0, 0.0],
            "vertical_axis": "Y",
            "floor_y": -float(camera_height),
            "units": "meters",
        },
        "floor_corners_3d": [
            {"x": float(x), "z": float(z)} for x, z in corners_xz
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def visualize(floor_xz: np.ndarray, processed_xz: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    closed = np.vstack([processed_xz, processed_xz[0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(floor_xz[:, 0], floor_xz[:, 1], s=3, alpha=0.18, label="raw floor_xz")
    ax.plot(closed[:, 0], closed[:, 1], "-o", linewidth=2.0, label="postprocessed")
    ax.scatter([0.0], [0.0], marker="+", s=100, c="red", label="camera")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    camera_height: float = 1.6,
    layout_height: float = 2.8,
    need_cube: bool = False,
) -> dict[str, Path]:
    input_path = Path(input_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    image_name = Path(data.get("image_name", input_path.stem)).stem

    if output_dir is None:
        output_dir = input_path.parent / "lgt_outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    floor_xz = boundary_to_floor_xz(
        data["floor_boundary"],
        data["image_size"],
        camera_height=camera_height,
    )
    floor_xz = floor_xz[np.isfinite(floor_xz).all(axis=1)]

    processed_xz = fit_layout(floor_xz=floor_xz.copy(), need_cube=need_cube, show=False)
    processed_xz = ensure_ccw(processed_xz.astype(np.float64))

    paths = {
        "layout3d": output_dir / f"{image_name}_layout3d.json",
        "floor_xz": output_dir / f"{image_name}_floor_xz.npy",
        "lgt_depth": output_dir / f"{image_name}_lgt_depth.npy",
        "processed_xz": output_dir / f"{image_name}_processed_xz.npy",
        "viz": output_dir / f"{image_name}_layout3d.png",
    }

    # np.save(paths["floor_xz"], floor_xz.astype(np.float32))
    # np.save(paths["processed_xz"], processed_xz.astype(np.float32))
    # np.save(paths["lgt_depth"], (np.linalg.norm(floor_xz, axis=1) / camera_height)[None].astype(np.float32))
    save_layout(processed_xz, paths["layout3d"], camera_height, layout_height)
    visualize(floor_xz, processed_xz, paths["viz"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw boundary JSON to post-processed layout3d files."
    )
    parser.add_argument("input", help="Raw boundary JSON path")
    parser.add_argument("-o", "--output-dir", default="outputs/3_layouts/")
    parser.add_argument("--camera-height", type=float, default=1.6)
    parser.add_argument("--layout-height", type=float, default=2.8)
    parser.add_argument("--need-cube", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run(
        args.input,
        output_dir=args.output_dir,
        camera_height=args.camera_height,
        layout_height=args.layout_height,
        need_cube=args.need_cube,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
