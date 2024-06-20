# from collections import Sequence
from typing import Sequence, Callable, Any, Optional
from monai.data import MetaTensor
import torch
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import convert_to_dst_type
from torch.cuda.amp import autocast
from torch.nn import functional as F
import numpy as np
from inference.utils import generate_point_prompt

def prepare_sam_val_input(
        inputs: MetaTensor,
        class_prompts: list[int] | torch.Tensor, point_prompts: dict[str, list],
        start_idx: int,
        original_affine: Optional[MetaTensor] = None, device=None, sam_image_size: int = 512):
    # Don't exclude background in val but will ignore it in metric calculation
    H, W = inputs.shape[1:]
    foreground_all = point_prompts["foreground"]
    background_all = point_prompts["background"]

    class_list = [[i + 1] for i in class_prompts]
    unique_labels = torch.tensor(class_list).long()
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        unique_labels = unique_labels.cuda()

    volume_point_coords = [cp for cp in foreground_all]
    volume_point_labels = [1] * len(foreground_all)

    for cp in background_all:
        volume_point_coords.append(cp)
        volume_point_labels.append(0)

    point_coords: list[list] = [[]]
    point_labels: list[list] = [[]]

    # Reoriente point coord if not in RAS
    if original_affine is not None:
        IJK2orientation = np.diag(original_affine[:3, :3])
        negative_indices = np.where(IJK2orientation < 0)[0]
        if len(negative_indices) > 0:
            for idx, c in enumerate(volume_point_coords):
                volume_point_coords[idx][negative_indices[0]] = H - volume_point_coords[idx][negative_indices[0]]
                volume_point_coords[idx][negative_indices[1]] = W - volume_point_coords[idx][negative_indices[1]]

    for idx, cp in enumerate(volume_point_coords):
        if cp[2] + 4 == start_idx:
            new_H = cp[0] * (sam_image_size / H)
            new_W = cp[1] * (sam_image_size / W)
            point_coords[0].append([new_H, new_W])
            point_labels[0].append(volume_point_labels[idx])

    if len(point_coords[0]) == 0:
        point_coords = None
        point_labels = None

    prepared_input = [{"image": inputs, "original_size": tuple(inputs.shape[1:])}]

    if len(class_prompts) == 0:
        class_enabled = False
    else:
        class_enabled = True
    if class_enabled:
        prepared_input[0].update({"labels": unique_labels})

    if point_coords:
        point_coords = torch.tensor(point_coords).long()
        point_labels = torch.tensor(point_labels).long()
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            point_coords = point_coords.cuda()
            point_labels = point_labels.cuda()

        prepared_input[0].update({"point_coords": point_coords, "point_labels": point_labels})

    return prepared_input, unique_labels


def vista_slice_inference(
        inputs: torch.Tensor | MetaTensor,
        predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        device: torch.device | str | None = None,
        n_z_slices: int = 9,
        labels: torch.Tensor | MetaTensor = None,
        *args: Any,
        **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)

    # labels = kwargs.pop("labels")
    num_classes = len(labels)

    inputs_l = inputs  # 1 x 1 x H x W x S
    # 1 x 11 x H x W x S
    pred_volume = torch.repeat_interleave(torch.zeros_like(inputs_l), num_classes + 1, dim=1).float()

    inputs_l = inputs_l.squeeze()  # 1 x H x W x S
    n_z_before_pad = inputs_l.shape[-1]
    # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice

    pd = (n_z_slices // 2, n_z_slices // 2)
    inputs_l = F.pad(inputs_l, pd, "constant", 0)  # 1 x H x W x (S + 2z)

    computeEmbedding = kwargs.pop("computeEmbedding")

    if computeEmbedding:
        embedding = compute_embedding(n_z_slices, n_z_before_pad, inputs_l, predictor)
        return embedding

    post_pred = Compose([Activations(sigmoid=True)])
    post_pred_slice = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    class_prompts = kwargs.pop("class_prompts")
    point_prompts = kwargs.pop("point_prompts")
    cached_data = kwargs.pop("cached_data")
    cached_pred = cached_data["pred"] if cached_data else None

    cachedEmbedding = kwargs.pop("cachedEmbedding")
    cachedEmbedding = cachedEmbedding if cachedEmbedding else None
    original_affine = kwargs.pop("original_affine")

    if (class_prompts is None) and (point_prompts is None):
        # Everything button: no class, no point prompts: iterate all slices
        unique_labels = torch.unique(labels)
        batch_labels_ = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=1).float()
        point_coords, point_labels = generate_point_prompt(batch_labels_, args)
        bg_labels = point_labels[point_labels == 0]
        bg_coords = point_labels[point_labels == 0]
        fg_labels = point_labels[point_labels == 1]
        fg_coords = point_labels[point_labels == 1]


        class_prompts = [i for i in range(num_classes)]
        point_prompts = {"foreground": list(fg_coords), "background": list(bg_coords)}
        pred_volume = iterate_all(
            pred_volume,  # 1 x 11 x H x W x S
            n_z_slices,
            n_z_before_pad,
            inputs_l,
            class_prompts,
            point_prompts,
            predictor,
            post_pred,
            cachedEmbedding,
            cached_pred,
            device,
        )
    elif (point_prompts is None) and (class_prompts is not None):
        if class_prompts:
            # class prompts only: need to iterate all slices
            point_prompts = {"foreground": [], "background": []}
            pred_volume = iterate_all(
                pred_volume,
                n_z_slices,
                n_z_before_pad,
                inputs_l,
                class_prompts,
                point_prompts,
                predictor,
                post_pred,
                cachedEmbedding,
                cached_pred,
                device,
            )
        else:
            pred_volume = pred_volume.argmax(1).unsqueeze(1)
    elif (class_prompts is None) and (point_prompts is not None):
        class_prompts = []
        pred_volume = update_slice(
            pred_volume,
            n_z_slices,
            n_z_before_pad,
            inputs_l,
            class_prompts,
            point_prompts,
            predictor,
            post_pred_slice,
            cached_pred,
            num_classes,
            original_affine,
            device,
        )
    else:
        pred_volume = update_slice(
            pred_volume,
            n_z_slices,
            n_z_before_pad,
            inputs_l,
            class_prompts,
            point_prompts,
            predictor,
            post_pred_slice,
            cached_pred,
            num_classes,
            original_affine,
            device,
        )

    if temp_meta is not None:
        final_output = convert_to_dst_type(pred_volume, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(pred_volume, inputs, device=device)[0]

    return final_output  # type: ignore


def compute_embedding(n_z_slices, n_z_before_pad, inputs_l, predictor):
    # image_embedding_dict saves the image embedding for each slice.
    # The key (int) is the index of center slice in original volume (before padding), e.g., 0,1,2,...n if the
    # original volume has n slices.
    # The value (torch.tensor) is the corresponding image embedding.
    image_embedding_dict = {}
    # get image embedding from the predictor (network) forward function
    for start_idx in range((n_z_slices // 2), (n_z_slices // 2 + n_z_before_pad)):
        inputs = inputs_l[..., start_idx - (n_z_slices // 2): start_idx + (n_z_slices // 2) + 1].permute(2, 0, 1)
        # Here, the batch size is 1 (it is possible to increase batch size if the device has enough memory).
        data = [{"image": inputs}]
        with autocast():
            image_embeddings = predictor.get_image_embeddings(data)  # (1, C, H, W)
        # Save image embedding for each slice to RAM
        image_embedding_dict[start_idx - (n_z_slices // 2)] = image_embeddings.cpu()

    return image_embedding_dict


def update_slice(
        pred_volume,
        n_z_slices,
        n_z_before_pad,
        inputs_l,
        class_prompts,
        point_prompts,
        predictor,
        post_pred_slice,
        cached_pred,
        num_classes,
        original_affine,
        device,
):
    z_indices = [p[2] + (9 // 2) for p in point_prompts["foreground"]]
    z_indices.extend([p[2] + (9 // 2) for p in point_prompts["background"]])
    z_indices = list(set(z_indices))

    pred_volume = pred_volume.argmax(1).unsqueeze(1)

    for start_idx in z_indices:
        if start_idx < (n_z_slices // 2):
            continue

        inputs = inputs_l[..., start_idx - (n_z_slices // 2): start_idx + (n_z_slices // 2) + 1].permute(2, 0, 1)
        if device and (device == "cuda" or isinstance(device, torch.device) and device.type == "cuda"):
            inputs = inputs.cuda()
        data, unique_labels = prepare_sam_val_input(
            inputs, class_prompts, point_prompts, start_idx, original_affine, device=device
        )

        predictor.eval()
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            with torch.cuda.amp.autocast():
                outputs = predictor(data)
                logit = outputs[0]["high_res_logits"]
        else:
            with torch.cpu.amp.autocast():
                outputs = predictor(data)
                logit = outputs[0]["high_res_logits"]

        out_list = torch.unbind(logit, dim=0)
        y_pred = torch.stack(post_pred_slice(out_list)).float()

        pred_volume = pred_volume.float()
        idx = torch.where(y_pred[0] == 1)
        z_idx = start_idx - (n_z_slices // 2)

        if cached_pred is not None:
            if class_prompts:
                cached_pred_idx = torch.where(cached_pred[:, :, :, z_idx] == class_prompts[0] + 1)
                cached_pred[:, :, :, z_idx][cached_pred_idx] = 0
                cached_pred[:, :, :, z_idx][idx] = class_prompts[0] + 1
            else:
                cached_pred[:, :, :, z_idx][idx] = num_classes + 1
        else:
            pred_volume[0, :, :, :, z_idx][idx] = class_prompts[0] + 1 if class_prompts else num_classes + 1

    if cached_pred is not None:
        pred_volume[0] = cached_pred.float()

    return pred_volume


def iterate_all(
        pred_volume,    # 1 x 11 x H x W x S
        n_z_slices,
        n_z_before_pad,
        inputs_l,       # 1 x H x W x (S + 2z)
        class_prompts,
        point_prompts,
        predictor,
        post_pred,
        cachedEmbedding,
        cached_pred,
        device,
):
    start_range = (
        range(n_z_slices // 2, min((n_z_slices // 2 + n_z_before_pad), len(cachedEmbedding)))
        if cachedEmbedding
        else range(n_z_slices // 2, n_z_slices // 2 + n_z_before_pad)
    )
    for start_idx in start_range:
        inputs = inputs_l[..., start_idx - n_z_slices // 2: start_idx + n_z_slices // 2 + 1].permute(2, 0, 1)
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            inputs = inputs.cuda()
        data, unique_labels = prepare_sam_val_input(inputs, class_prompts, point_prompts, start_idx, device=device)
        predictor = predictor.eval()
        with autocast():
            if cachedEmbedding:
                curr_embedding = cachedEmbedding[start_idx]
                if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                    curr_embedding = curr_embedding.cuda()
                outputs = predictor.get_mask_prediction(data, curr_embedding)
            else:
                outputs = predictor(data)
            logit = outputs[0]["high_res_logits"]
            # print(f'logit shape: {logit.shape}')

        out_list = torch.unbind(logit, dim=0)
        y_pred = torch.stack(post_pred(out_list)).float()
        # print(f'y_pred shape: {y_pred.shape}')
        pred_idx = start_idx - (n_z_slices // 2) if not cachedEmbedding else start_idx
        # 1 x 11 x H x W x (S + 2z)
        pred_volume[0, unique_labels, ..., pred_idx] = y_pred
    # print('pred_volume.shape before argmax and unsqueeze: ', pred_volume.shape)
    pred_volume = pred_volume.argmax(1).unsqueeze(1).cpu()
    # print(f'pred_volume.shape become: {pred_volume.shape}')
    pred_volume = pred_volume.float()

    if cached_pred is not None:
        pred_volume_idx = torch.where(pred_volume[0] != 0)
        cached_pred[pred_volume_idx] = pred_volume[0][pred_volume_idx]
        pred_volume[0] = cached_pred.float()

    return pred_volume
