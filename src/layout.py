"""Standalone LGT/DUA-style Manhattan floor-plan post-processing."""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_floorplan(
    xz: np.ndarray,
    fill_color: list[float] | None = None,
    border_color: list[float] | None = None,
    side_l: int = 512,
    show_radius: float | None = None,
    show: bool = False,
    marker_color: list[float] | None = None,
    center_color: list[float] | None = None,
    scale: float = 1.5,
) -> np.ndarray:
    """Rasterize an X/Z polygon to a square top-down mask."""
    if fill_color is None:
        fill_color = [1]

    board = np.zeros([side_l, side_l, len(fill_color)], dtype=float)
    xz = np.asarray(xz, dtype=np.float64).copy()
    if show_radius is None:
        show_radius = float(np.linalg.norm(xz, axis=-1).max())
    show_radius = max(show_radius, 1e-6)

    xz = xz * side_l / (2 * scale) / show_radius
    xz[:, 1] = -xz[:, 1]
    xz += side_l // 2
    xz = xz.astype(int)
    cv2.fillPoly(board, [xz], fill_color)
    if border_color:
        cv2.drawContours(board, [xz], 0, border_color, 2)
    if marker_color is not None:
        for point in xz:
            cv2.drawMarker(board, tuple(point), marker_color, markerType=0, markerSize=10, thickness=2)
    if center_color is not None:
        cv2.drawMarker(board, (side_l // 2, side_l // 2), center_color, markerType=0, markerSize=10, thickness=2)
    if show:
        plt.axis("off")
        plt.imshow(board[..., 0] if board.shape[-1] == 1 else board)
        plt.show()
    return board


def calc_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm <= 1e-9:
        return 0.0
    value = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
    return float(np.arccos(value))


def merge_near(values: list[list[int]], diag: float, min_value: int) -> list[int]:
    group = [[min_value]]
    for i in range(1, len(values)):
        if values[i][1] == 0 and values[i][0] - np.mean(group[-1]) < diag * 0.02:
            group[-1].append(values[i][0])
        else:
            group.append([values[i][0]])
    if len(group) == 1:
        return [values[0][0], values[-1][0]]
    return [int(np.mean(item)) for item in group]


def _find_contours(mask: np.ndarray, mode: int, method: int) -> list[np.ndarray]:
    contours = cv2.findContours(mask, mode, method)
    if len(contours) == 3:
        contours = contours[1]
    else:
        contours = contours[0]
    return list(contours)


def fit_layout(
    floor_xz: np.ndarray,
    need_cube: bool = False,
    show: bool = False,
    block_eps: float = 5,
) -> np.ndarray:
    """Fit a Manhattan polygon to dense floor X/Z points.

    This is a standalone adaptation of LGT-Net's optimized DuLa post-processing.
    """
    floor_xz = np.asarray(floor_xz, dtype=np.float64)
    show_radius = float(np.linalg.norm(floor_xz, axis=-1).max())
    side_l = 512
    floorplan = draw_floorplan(
        xz=floor_xz,
        show_radius=show_radius,
        show=show,
        scale=1,
        side_l=side_l,
    ).astype(np.uint8)
    center = np.array([side_l / 2, side_l / 2])

    polys = _find_contours(floorplan, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not polys:
        raise ValueError("Cannot find a floor-plan contour from floor_xz")
    polys.sort(key=cv2.contourArea, reverse=True)
    poly = polys[0]

    sub_x, sub_y, width, height = cv2.boundingRect(poly)
    floorplan_sub = floorplan[sub_y : sub_y + height, sub_x : sub_x + width]
    sub_center = center - np.array([sub_x, sub_y])

    polys = _find_contours(floorplan_sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not polys:
        raise ValueError("Cannot find a contour in cropped floor-plan")
    poly = polys[0]
    epsilon = 0.005 * cv2.arcLength(poly, True)
    poly = cv2.approxPolyDP(poly, epsilon, True)

    x_lst = [[int(poly[:, 0, 0].min()), 0]]
    y_lst = [[int(poly[:, 0, 1].min()), 0]]
    ans = np.zeros((floorplan_sub.shape[0], floorplan_sub.shape[1]))

    for i in range(len(poly)):
        p1 = poly[i][0]
        p2 = poly[(i + 1) % len(poly)][0]
        cp1 = p1 - sub_center
        cp2 = p2 - sub_center
        p12 = p2 - p1
        l1 = np.linalg.norm(cp1)
        l2 = np.linalg.norm(cp2)

        is_block1 = np.rad2deg(calc_angle(cp1, cp2)) < block_eps
        is_block2 = np.rad2deg(calc_angle(cp2, p12)) < block_eps * 2
        is_block3 = np.rad2deg(calc_angle(cp2, -p12)) < block_eps * 2
        is_block = is_block1 and (is_block2 or is_block3)

        slope = 10 if (p2[0] - p1[0]) == 0 else abs((p2[1] - p1[1]) / (p2[0] - p1[0]))

        if is_block:
            y_lst.append([int(p1[1] if l1 < l2 else p2[1]), 1])
            x_lst.append([int(p1[0] if l1 < l2 else p2[0]), 1])

            left, right = sorted([p1[0], p2[0]])
            top, bottom = sorted([p1[1], p2[1]])
            sample = floorplan_sub[top:bottom, left:right]
            score = 0 if sample.size == 0 else sample.mean()
            if score >= 0.3:
                ans[top:bottom, left:right] = 1
        elif slope <= 1:
            y_lst.append([int((p1[1] + p2[1]) / 2), 0])
        else:
            x_lst.append([int((p1[0] + p2[0]) / 2), 0])

    x_lst.append([int(poly[:, 0, 0].max()), 0])
    y_lst.append([int(poly[:, 0, 1].max()), 0])
    x_lst.sort(key=lambda item: item[0])
    y_lst.sort(key=lambda item: item[0])

    diag = math.sqrt(floorplan_sub.shape[1] ** 2 + floorplan_sub.shape[0] ** 2)
    x_values = merge_near(x_lst, diag, int(poly[:, 0, 0].min()))
    y_values = merge_near(y_lst, diag, int(poly[:, 0, 1].min()))
    if need_cube and len(x_values) > 2:
        x_values = [x_values[0], x_values[-1]]
    if need_cube and len(y_values) > 2:
        y_values = [y_values[0], y_values[-1]]

    for i in range(len(x_values) - 1):
        for j in range(len(y_values) - 1):
            sample = floorplan_sub[y_values[j] : y_values[j + 1], x_values[i] : x_values[i + 1]]
            score = 0 if sample.size == 0 else sample.mean()
            if score >= 0.3:
                ans[y_values[j] : y_values[j + 1], x_values[i] : x_values[i + 1]] = 1

    pred = np.uint8(ans)
    pred_polys = _find_contours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not pred_polys:
        raise ValueError("Cannot find a post-processed layout contour")
    pred_polys.sort(key=cv2.contourArea, reverse=True)
    pred_poly = pred_polys[0]

    for i in range(len(pred_poly)):
        p1 = pred_poly[i][0]
        p2 = pred_poly[(i + 1) % len(pred_poly)][0]
        if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
            p1[0] = p2[0]
        else:
            p1[1] = p2[1]

    polygon = [(point[0][1], point[0][0]) for point in pred_poly[::-1]]
    v = np.array([point[0] + sub_y for point in polygon])
    u = np.array([point[1] + sub_x for point in polygon])
    pred_xz = np.concatenate(
        (u[:, np.newaxis] - side_l // 2, side_l // 2 - v[:, np.newaxis]),
        axis=1,
    )
    pred_xz = pred_xz * show_radius / (side_l // 2)

    pred_xz = clean_manhattan_polygon(pred_xz)

    if show:
        draw_floorplan(pred_xz, show_radius=show_radius, show=show)
    return pred_xz


def clean_manhattan_polygon(points: np.ndarray, short_edge: float = 0.08, tol: float = 1e-6) -> np.ndarray:
    """Remove tiny contour artifacts while preserving concave Manhattan corners."""
    points = remove_duplicate_neighbors(np.asarray(points, dtype=np.float64), eps=tol)

    for _ in range(3):
        points = snap_edges_to_axes(points)
        points = remove_duplicate_neighbors(points, eps=short_edge)
        points = remove_short_edges(points, min_length=short_edge)
        points = remove_collinear(points, tol=max(short_edge, tol))

    return ensure_ccw(points)


def snap_edges_to_axes(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    for i in range(len(points)):
        j = (i + 1) % len(points)
        dx = abs(points[j, 0] - points[i, 0])
        dz = abs(points[j, 1] - points[i, 1])
        if dx < dz:
            points[j, 0] = points[i, 0]
        else:
            points[j, 1] = points[i, 1]
    return points


def remove_short_edges(points: np.ndarray, min_length: float) -> np.ndarray:
    points = remove_duplicate_neighbors(points, eps=1e-9)
    changed = True
    while changed and len(points) > 4:
        changed = False
        for i in range(len(points)):
            j = (i + 1) % len(points)
            if np.linalg.norm(points[j] - points[i]) < min_length:
                points = np.delete(points, j, axis=0)
                changed = True
                break
    return points


def remove_collinear(points: np.ndarray, tol: float) -> np.ndarray:
    points = remove_duplicate_neighbors(points, eps=1e-9)
    changed = True
    while changed and len(points) > 4:
        changed = False
        keep = []
        for i in range(len(points)):
            prev_pt = points[i - 1]
            pt = points[i]
            next_pt = points[(i + 1) % len(points)]
            same_x = abs(prev_pt[0] - pt[0]) <= tol and abs(pt[0] - next_pt[0]) <= tol
            same_z = abs(prev_pt[1] - pt[1]) <= tol and abs(pt[1] - next_pt[1]) <= tol
            if same_x or same_z:
                changed = True
            else:
                keep.append(pt)
        if len(keep) >= 4:
            points = np.asarray(keep, dtype=np.float64)
        else:
            break
    return points


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
