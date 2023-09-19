import cv2
import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.cm as cm


def interpolate_depth(pos, depth):

    device = pos.device
    ids = torch.arange(0, pos.size(1), device=device)
    h, w = depth.size()
    
    i = pos[1, :]
    j = pos[0, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left
    
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([j.view(1, -1), i.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.moveaxis(image, -1, 0)
    #image = np.expand_dims(image, 0)
    if preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    else:
        pass
    return image


def draw_matches(img0, img1, kpts0, kpts1, matches, scores, margin=10, channels_first=True, draw_kpts=False):
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = scores[valid]
    color = cm.jet(mconf)
    if(channels_first):
        C0, H0, W0 = img0.shape
        C1, H1, W1 = img1.shape
    else:
        H0, W0, C0 = img0.shape
        H1, W1, C1 = img1.shape

    C, H, W = max(C0, C1), max(H0, H1), W0 + W1 + margin

    if(channels_first):
        out = np.ones((C, H, W))
        out[:, :H0, :W0] = img0
        out[:, :H1, W0+margin:] = img1
        out = np.swapaxes(out, 0, 2)
    else:
        out = np.ones((H, W, C))
        out[:H0, :W0, :] = img0
        out[:H1, W0+margin:, :] = img1
       
    #out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

    color = (np.array(color[:, :3])).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        if(draw_kpts):
            cv2.circle(out, (x0, y0), 2, color=(0,0,0), thickness=-1)
            cv2.circle(out, (x1 + margin + W0, y1), 2, color=(0,0,0), thickness=-1)
    return out


def scores_to_matches(scores, threshold=0.5):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    valid0 = mutual0 & (mscores0 > threshold)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    
    return indices0, mscores0


def pad_data(data, max_kpts, img_shape, device):
    _, _, width, _ = img_shape

    for k in data:
        if isinstance(data[k], (list, tuple)):
            new_data = []
            if(k.startswith('keypoints')):
                #padding keypoints
                for kpt in data[k]:
                    #random_values = torch.Tensor(max_kpts - kpt.shape[0], 2).uniform_(0, width)
                    random_values = torch.randint(0, width, (max_kpts - kpt.shape[0], 2))
                    new_data += [torch.cat((kpt, random_values.to(device)), 0)]
                    
            if(k.startswith('descriptor')):
                #padding descriptors
                for desc in data[k]:
                    new_data += [F.pad(desc, 
                                (0, max_kpts - desc.shape[1]))]

            if(k.startswith('score')):
                #padding scores
                for score in data[k]:
                    new_data += [F.pad(score, 
                                (0, max_kpts - score.shape[0]))]
            data[k] = torch.stack(new_data)
    return data
    

def replace_ignored(data, ignore, img_shape, device):
    _, _, width, _ = img_shape

    for img_id in ['0', '1']:
        for k in data:
            batch_size = data[k].size(0)
            if(k.startswith('keypoints'+img_id)):
                for i in range(batch_size):
                    for id in ignore['ignored'+img_id][i]:
                        new_row = torch.randint(0, width, (1, 2))
                        data[k][i][id] = new_row
            if(k.startswith('score'+img_id)):
                for i in range(batch_size):        
                    for id in ignore['ignored'+img_id][i]:
                        data[k][i][id] = 0
    return data


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def min_row_col(tensor):
    i = 0
    smallest, min_i, min_j = None, None, None
    for row in tensor:
        min_value = torch.min(row)
        if(smallest == None or min_value < smallest):
            smallest = min_value
            min_i = i
            min_j = torch.argmin(row).item()
        i += 1

    return min_i, min_j


def draw_patches(img, patches_corners, color=(0,0,0), thickness=2):
    for corner in patches_corners:
        img = cv2.rectangle(img, (int(corner[0][0]), int(corner[0][1])), 
                                 (int(corner[1][0]), int(corner[1][1])), color, thickness)
    return img

def draw_pts(img, pts, color=(255, 255, 255), radius=0, thickness=-1):
    for pt in pts:
        if(pt[0] > 0 and pt[1] > 0):
            img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=radius, color=color, thickness=thickness)
    return img


def patch_meshgrid(x_min, x_max, y_min, y_max):
    xs = torch.range(x_min, x_max-1, step=1)
    ys = torch.range(y_min, y_max-1, step=1)
    gx, gy = torch.meshgrid(xs, ys)
    grid = torch.cat((torch.unsqueeze(gx, dim=2), torch.unsqueeze(gy, dim=2)), dim=2)
    return grid


def get_only_balanced(data, gt, max_kpts):
    new_data = defaultdict(lambda: None)
    new_gt = None
    for i in range(gt.size(0)):
        valid_ids = (gt[i] != -1).nonzero(as_tuple=True)
        filtered_target = gt[i][valid_ids]
        pos_ids = (filtered_target < max_kpts).nonzero(as_tuple=True)
        neg_ids = (filtered_target == max_kpts).nonzero(as_tuple=True) 
        total_size = len(pos_ids[0])+len(neg_ids[0])
        
        if(len(pos_ids[0])/total_size > 0.5):
            if(new_gt == None):
                new_gt = torch.unsqueeze(gt[i], dim=0)
            else:
                new_gt = torch.cat((new_gt, torch.unsqueeze(gt[i], dim=0)), dim=0)
            
            for k in data:
                if(new_data[k] == None):
                    new_data[k] = torch.unsqueeze(data[k][i], dim=0)
                else:
                    new_data[k] = torch.cat((new_data[k], torch.unsqueeze(data[k][i], dim=0)), dim=0)
    return new_data, new_gt


def fill_dustbins(matches):
    rows = torch.count_nonzero(matches, dim=1)
    cols = torch.count_nonzero(matches, dim=0)
    dust_col = rows.clone()
    dust_row = cols.clone()
    dust_col[rows == 0] = 1
    dust_col[rows != 0] = 0
    dust_row[cols == 0] = 1
    dust_row[cols != 0] = 0
    matches[:,-1] = dust_col
    matches[-1,:] = dust_row
    return matches


def ohe_to_le(ohe_tensor):
    '''
        Function to convert one hot encoding to label encoding. Notice that if all elements in a row/cols are zero, the keypoint has no match, 
        thus its label is assigned to n_rows/n_cols. MOreover, if the keypoint is ignored its label is assigned to -1
    '''
    le_tensor = torch.full((ohe_tensor.size(0), ohe_tensor.size(-1)), ohe_tensor.size(-1))
    match_ids = (ohe_tensor == 1).nonzero(as_tuple=True)
    ignored_ids = (ohe_tensor == -1).nonzero(as_tuple=True)    
    le_tensor[match_ids[:2]] = match_ids[2]
    le_tensor[ignored_ids[:2]] = -1
    
    return le_tensor


def get_kpts_depths(kpts_batch, depth_batch, device):
    keypoint_depth_batch = torch.full((kpts_batch.size(0), kpts_batch.size(1)), 
                                        -1, device=device).float()
    for idx_in_batch in range(len(kpts_batch)):
        kpts = kpts_batch[idx_in_batch].to(device)
        kpts = torch.transpose(kpts, 0, 1)
        depths, _, ids = interpolate_depth(kpts, depth_batch[idx_in_batch].to(device))
        keypoint_depth_batch[idx_in_batch, ids] = depths/torch.max(depths)
    return keypoint_depth_batch


def save_model(path, model, optimizer, step, epoch, loss):
    torch.save({'epoch': epoch,
                'step': step,
                'kenc': model.kenc.state_dict(),
                'gnn': model.gnn.state_dict(),
                'final_proj': model.final_proj.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss}, 
                path)
    print(f'Model {path} saved!')


def load_model_weights(model, path, recover_state=False, modules=['gnn', 'final_proj']):
    print('Loading model ', path)
    ckpt = torch.load(str(path))
    if(modules):
        if('kenc' in modules):
            model.kenc.load_state_dict(ckpt['kenc'])
        if('gnn' in modules):
            model.gnn.load_state_dict(ckpt['gnn'])
        if('final_proj' in modules):
            model.final_proj.load_state_dict(ckpt['final_proj'])
    else:
        model.load_state_dict(ckpt)
    if(recover_state):
        return model, ckpt['epoch'], ckpt['step'], ckpt['optimizer'], ckpt['loss']
    return model


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
