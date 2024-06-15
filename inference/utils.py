import torch
import numpy as np
import random
from copy import deepcopy


def apply_coords_torch(coords: np.ndarray, original_size, sam_image_size) -> np.ndarray:
    """
    @param coords: shape with N x 2, 0: x, 1: y
    @param original_size:
    @param sam_image_size:
    @return:

    """
    old = original_size
    new = sam_image_size
    coords = deepcopy(coords).float()
    # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
    # longest side then resize it to sam_image_size. In other words, the scale factor is determined by the longest side.
    coords[..., 0] = coords[..., 0] * (new / old)
    coords[..., 1] = coords[..., 1] * (new / old)
    return coords


def sample_points(labelpoints, n_points):
    idx = torch.randperm(len(labelpoints), dtype=torch.long, device=labelpoints.device)[:n_points]
    return [labelpoints[idx]]


def generate_point_prompt(batch_labels_, args, points_pos=None, points_neg=None, previous_pred=None, grid=False):
    max_point = args.max_points

    if points_pos is not None:
        Np = points_pos
    else:
        gauss_random = random.gauss(mu=0, sigma=max_point // 2)
        Np = min(max_point, int(np.abs(gauss_random)) + 1)

    if points_neg is not None:
        Nn = points_neg
    else:
        gauss_random = random.gauss(mu=0, sigma=max_point // 2)
        Nn = min(max_point, int(np.abs(gauss_random)))

    # To follow original SAM, with equal probability either a foreground point
    # is selected randomly for the target mask
    _point = []
    _point_label = []
    b, h, w = batch_labels_.shape
    device = batch_labels_.device
    for i in range(b):
        plabels = batch_labels_[i, ...]
        nlabels = (plabels == 0.0).float()

        if previous_pred is not None:
            ppred = previous_pred[i, 0, ...]
            npred = (previous_pred[i, 0, ...] == 0.0).float() # background

            # False positive mask (pixels that are predicted as positive but are actually negative)
            fp_mask = torch.logical_and(nlabels, ppred)
            # False negative mask (pixels that are predicted as negative but are actually positive)
            fn_mask = torch.logical_and(plabels, npred)
            # we sample positive points from false negative pred.
            # we sample negative points from false positive pred.
            plabelpoints = torch.nonzero(fn_mask)
            nlabelpoints = torch.nonzero(fp_mask)

        else:
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
        # 1 indicates a foreground point and 0 indicates a background point.
        # -1 indicates a dummy non-point as the placeholder.
        n_placeholder = Np + Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn)

        # Use torch.randperm to generate indices on a GPU tensor
        _point.append(
            torch.cat(
                sample_points(plabelpoints, min(len(plabelpoints), Np))
                + sample_points(nlabelpoints, min(len(nlabelpoints), Nn))
                + [torch.zeros((1, 2), device=device)] * n_placeholder,
                dim=0,
            )
        )

        pos_label_prompt = [1] * min(len(plabelpoints), Np)
        neg_label_prompt = [0] * min(len(nlabelpoints), Nn)
        placeholder = [-1] * n_placeholder
        _point_label.append(
            torch.tensor(pos_label_prompt + neg_label_prompt + placeholder).to(device)
        )

    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    point_coords = apply_coords_torch(point, max(h, w), args.sam_image_size)

    return point_coords, point_label


def prepare_slice_data(image, args):
    unique_labels = torch.tensor([i for i in range(1, args.nc)]).cuda(args.rank)
    labels_prompt = unique_labels.unsqueeze(-1)
    prepare_input = [{'image': image, 'original_size': tuple(image.shape[-2:]), 'labels': labels_prompt}]

    return prepare_input, unique_labels

if __name__ == '__main__':
    args = object()
    args.__setattr__('max_points', 30)
    args.__setattr__('nc', 11)
    kwargs = {
        'points_pos': 12,
        'points_neg': 12,

    }
    unique_labels = torch.tensor([i for i in range(1, args.nc)])
    labels = torch.randint(0, 11, (64, 64))

    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    out = generate_point_prompt(batch_labels, args)