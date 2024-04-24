import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from gluefactory.geometry.depth import sample_depth, project
from gluefactory.utils.tensor import batch_to_device
from gluefactory.visualization.viz2d import plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches, cm_RdGn
from gluefactory.geometry.wrappers import Pose

def normalize_image(image):
    """
    Normalize using imagenet mean and std
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def get_sorted_matches(data):
    """
    Find matches ordered by score
    """
    
    m0 = data["matches0"]
    scores0 = data["matching_scores0"]
    b_size = m0.size(0)
    b_mcfs = torch.empty(b_size, m0.size(1), 3)

    for i in range(b_size):
        mcfs = torch.empty(m0.size(1), 3)
        for j, (m, c) in enumerate(zip(m0[i], scores0[i])):
            mcfs[j] = torch.tensor([j, m, c])  
        sorted_indices = torch.argsort(mcfs[:, 2], descending=True)
        sorted_mcfs = mcfs[sorted_indices]
        b_mcfs[i] = sorted_mcfs
    return b_mcfs
    

def debug_batch(data, n_pairs=2):
    '''
    Visualize the first n_pairs in the batch
    Copied from gluefactory.visualization.visualize_batch.py
    '''
    if "0to1" in data.keys():
        data = data["0to1"]
    
    data = batch_to_device(data, "cpu", non_blocking=False)
    images, kpts, matches, mcolors, images_projs = [], [], [], [], []
    heatmaps = []
    view0, view1 = data["view0"], data["view1"]
    pred = data["pred_vo"]
    R_pred = euler_angles_to_matrix(pred[..., 3:], "XYZ")
    t_pred = pred[..., :3]
    T_0to1_pred = Pose.from_Rt(R_pred, t_pred)
    T_1to0_pred = T_0to1_pred.inv()

    n_pairs = min(n_pairs, view0["image"].shape[0])
    
    assert view0["image"].shape[0] >= n_pairs
    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    m0 = data["matches0"]
    
    for i in range(n_pairs):
        valid = (m0[i] > -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = m0[i][valid]

        if "heatmap0" in data.keys():
            heatmaps.append(
                [
                    torch.sigmoid(data["heatmap0"][i, 0]),
                    torch.sigmoid(data["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())
        
        # kpts0_1 = get_kpts_projection(kp0[i][valid], view0["depth"][i], view0["camera"], view1["camera"], T_0to1_pred)
        # img_proj0 = draw_patches(images[i][1], kpts0_1[0], color=(0,255,0))
        #img_proj1 = draw_patches(images[i][0], kpts1_0[0], color=(0,255,0))
        #images_projs.append([img_proj0,img_proj1])

    fig_matches, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
       [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    #fig_projs, axes = plot_image_grid(images_projs, return_fig=True, set_lim=True)   
    
    return fig_matches, None

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

def draw_matches(image0, image1, kpts0, kpts1, scores=None):
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
    return out


def get_patches(img, pts, patch_size=16):
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