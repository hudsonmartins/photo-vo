import torch
import cv2
import numpy as np
from gluefactory.geometry.depth import sample_depth, project
from gluefactory.visualization.visualize_batch import make_match_figures
from gluefactory.utils.tensor import batch_to_device
from gluefactory.visualization.viz2d import plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches, cm_RdGn


def get_sorted_kpts_by_matches(data):
    """
    Sort keypoints and matches by matching scores
    """
    kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
    m0 = data["matches0"]
    scores0 = data["matching_scores0"]
    bsize = kpts0.size(0)
    skpts0, skpts1, sm0, sscores0 = [], [], [], []

    for i in range(bsize):
        valid0 = m0[i] > -1
        kpts0_valid = kpts0[i][valid0].detach()
        kpts1_valid = kpts1[i][m0[i][valid0]].detach()
        m0_valid = m0[i][valid0].detach()
        scores0_valid = scores0[i][valid0].detach()
        #sort by matching scores
        scores0_valid, indices = torch.sort(scores0_valid, descending=True)
        kpts0_valid = kpts0_valid[indices]
        kpts1_valid = kpts1_valid[indices]
        m0_valid = m0_valid[i][indices]
        skpts0.append(kpts0_valid)
        skpts1.append(kpts1_valid)
        sm0.append(m0_valid)
        sscores0.append(scores0_valid)

    data["sorted_keypoints0"] = torch.Tensor(np.array(skpts0))
    data["sorted_keypoints1"] = torch.Tensor(np.array(skpts1))
    data["sorted_matches0"] = torch.Tensor(np.array(sm0))
    data["sorted_matching_scores0"] = torch.Tensor(np.array(sscores0))

    return data



def debug_batch(data, pred, n_pairs=2):
    '''
    Visualize the first n_pairs in the batch
    Copied from gluefactory.visualization.visualize_batch.py
    '''
    if "0to1" in pred.keys():
        pred = pred["0to1"]
    
    data = batch_to_device(data, "cpu", non_blocking=False)
    pred = batch_to_device(pred, "cpu", non_blocking=False)
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    
    assert view0["image"].shape[0] >= n_pairs
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    
    for i in range(n_pairs):
        valid = (m0[i] > -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
       [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]
    return fig

def get_kpts_projection(kpts, depth, camera0, camera1, T_0to1):
    d, valid = sample_depth(kpts, depth)
    kpts = kpts * valid.unsqueeze(-1)

    kpts_1, visible = project(
        kpts, d, depth, camera0, camera1, T_0to1, valid
    )
    kpts = kpts * visible.unsqueeze(-1)
    kpts_1 = kpts_1 * visible.unsqueeze(-1)
    kpts[~visible] = float('nan')
    kpts_1[~visible] = float('nan')
    return kpts, kpts_1

    
def draw_patches(img, kpts, color=(0,0,255), patch_size=10):
    """Draw patches around keypoints on an image"""
    half = patch_size // 2
    for pt in kpts:
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        img = cv2.rectangle(
            np.ascontiguousarray(img).astype(np.uint8),
            (int(pt[0]) - half, int(pt[1]) - half),
            (int(pt[0]) + half, int(pt[1]) + half),
            color,
            2,
        )
    return img


def draw_pts(img, pts, color=(0, 0, 255), radius=5):
    """Draw points on an image"""
    for pt in pts:
        #check if pt is nan
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        img = cv2.circle(np.ascontiguousarray(img).astype(np.uint8), (int(pt[0]), int(pt[1])), radius, color, 2)
    return img



def get_patches(img, pts, patch_size=10):
    """Given an image and a set of points, return the patches around the points"""

    batch_size = img.size(0)
    patch_size = int(patch_size)
    half = patch_size // 2
    patches = []
    for i in range(batch_size):
        #pad image
        img_pad = torch.nn.functional.pad(img[i], (half, half, half, half), mode='reflect')
        pts_i = pts[i]
        patches_i = []
        for j in range(pts_i.size(0)):
            if torch.isnan(pts_i[j]).any():
                #append nan tensor
                patches_i.append(torch.full((3, patch_size, patch_size), float('nan')))
                continue
            x, y = pts_i[j].int()            
            patch = img_pad[..., y:y+patch_size, x:x+patch_size]
            patches_i.append(patch)
        patches.append(torch.stack(patches_i))

    return torch.stack(patches)

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Extracted from https://github.com/facebookresearch/pytorch3d
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Extracted from https://github.com/facebookresearch/pytorch3d
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extracted from https://github.com/facebookresearch/pytorch3d
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    """
    Extracted from https://github.com/facebookresearch/pytorch3d
    """
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Extracted from https://github.com/facebookresearch/pytorch3d
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
