"""외벽 평면 → 박스형 건물 메시 (trimesh). 보이는 면 수에 따라
  1면 → 두께 있는 벽 판, 2면 직교 → L자 (두 판이 모서리에서 만남), 2면 평행·3~4면 → 박스.
FacadeFrame의 origin·width를 메시 모서리에 맞게 고친 새 목록을 돌려준다 (텍스처·창문 좌표는 이것 기준).
"""

from dataclasses import replace

import numpy as np
import trimesh

from facade_cv.recon3d.geometry import UP, FacadeFrame


def _slab(f: FacadeFrame, thickness: float) -> trimesh.Trimesh:
    """외벽 사각형에서 안쪽(-normal)으로 thickness만큼의 판."""
    T = np.eye(4)
    T[:3, :3] = np.stack([f.u_axis, f.v_axis, f.normal], 1)
    T[:3, 3] = f.origin + f.u_axis * f.width / 2 + f.v_axis * f.height / 2 - f.normal * thickness / 2
    return trimesh.creation.box(extents=(f.width, f.height, thickness), transform=T)


def _side_span(f: FacadeFrame, axis: np.ndarray) -> tuple[float, float]:
    ends = [f.origin @ axis, (f.origin + f.u_axis * f.width) @ axis]
    return min(ends), max(ends)


def _reframe(f: FacadeFrame, p0: np.ndarray, p1: np.ndarray, ground: float, height: float) -> FacadeFrame:
    """수평 끝점 p0, p1 (평면 위) 사이로 외벽 범위를 다시 잡는다. origin = u가 작은 쪽."""
    a, b = sorted([p0, p1], key=lambda p: p @ f.u_axis)
    origin = np.array([a[0], ground, a[2]])
    return replace(f, origin=origin, width=float((b - a) @ f.u_axis), height=height)


def build_mesh(facades: list[FacadeFrame], cfg: dict) -> tuple[trimesh.Trimesh, list[FacadeFrame], str]:
    """(메시, 메시에 맞춘 facade 목록, kind: slab | L | box)."""
    if not facades:
        raise ValueError("no facades")
    ground = min(f.origin[1] for f in facades)
    height = max(f.origin[1] + f.height for f in facades) - ground
    t = cfg["wall_thickness_ratio"]

    if len(facades) == 1:
        f = replace(facades[0], origin=facades[0].origin * [1, 0, 1] + [0, ground, 0], height=height)
        return _slab(f, min(f.width, f.height) * t), [f], "slab"

    n0 = facades[0].normal
    e1 = n0 / np.linalg.norm(n0)
    e3 = np.cross(e1, UP)  # 박스 수평 두 축 (e1, e3)
    perpendicular = len(facades) == 2 and abs(facades[0].normal @ facades[1].normal) < 0.5

    if perpendicular:  # L자: 두 평면의 수직 교선 = 모서리. 각 면을 모서리까지 늘리거나 줄인다
        a, b = facades
        A = np.array([[a.normal[0], a.normal[2]], [b.normal[0], b.normal[2]]])
        x, z = np.linalg.solve(A, [-a.offset, -b.offset])
        corner = np.array([x, ground, z])
        out = []
        for f in (a, b):
            ends = [f.origin, f.origin + f.u_axis * f.width]
            far = max(ends, key=lambda p: np.linalg.norm((p - corner) * [1, 0, 1]))
            out.append(_reframe(f, corner, far, ground, height))
        mesh = trimesh.util.concatenate([_slab(f, min(f.width, f.height) * t) for f in out])
        return mesh, out, "L"

    # 박스: 각 축의 양 끝 = 그 방향 외벽 평면, 없으면 다른 외벽들의 가로 범위 끝
    bounds = {}
    for axis_name, axis in (("e1", e1), ("e3", e3)):
        spans = [_side_span(f, axis) for f in facades if abs(f.normal @ axis) < 0.5]
        lo_hi = []
        for sign in (-1, 1):
            planes = [f for f in facades if f.normal @ axis * sign > 0.5]
            if planes:
                lo_hi.append(-planes[0].offset * sign)  # n·x = -d, n = sign·axis → axis·x = -d·sign
            elif spans:
                lo_hi.append(min(s[0] for s in spans) if sign < 0 else max(s[1] for s in spans))
            else:
                raise ValueError(f"cannot bound box along {axis_name}")
        bounds[axis_name] = lo_hi
    (lo1, hi1), (lo3, hi3) = bounds["e1"], bounds["e3"]
    T = np.eye(4)
    T[:3, :3] = np.stack([e1, UP, e3], 1)
    T[:3, 3] = e1 * (lo1 + hi1) / 2 + e3 * (lo3 + hi3) / 2 + UP * (ground + height / 2)
    mesh = trimesh.creation.box(extents=(hi1 - lo1, height, hi3 - lo3), transform=T)

    def pt(c1: float, c3: float) -> np.ndarray:
        return e1 * c1 + e3 * c3

    out = []
    for f in facades:
        if abs(f.normal @ e1) > 0.5:
            c1 = hi1 if f.normal @ e1 > 0 else lo1
            out.append(_reframe(f, pt(c1, lo3), pt(c1, hi3), ground, height))
        else:
            c3 = hi3 if f.normal @ e3 > 0 else lo3
            out.append(_reframe(f, pt(lo1, c3), pt(hi1, c3), ground, height))
    return mesh, out, "box"
