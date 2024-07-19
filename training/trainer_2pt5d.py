# Copyright 2020 - 2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import time
from copy import deepcopy
from argparse import Namespace
from vista_2pt5d.model import sam_model_registry, Vista2pt5D

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import decollate_batch, MetaTensor
from monai.metrics import compute_dice
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import wandb

def apply_coords_torch(coords, original_size, sam_image_size) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
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


def generate_point_prompt(batch_labels_, args, points_pos=None, points_neg=None, previous_pred=None):
    """

    @param batch_labels_:
    @param args:
    @param points_pos:
    @param points_neg:
    @param previous_pred:
    @return:
    """
    max_point = args.max_points
    if points_pos is not None:
        Np = points_pos
    else:
        gauss = random.gauss(mu=0, sigma=max_point // 2)
        gauss_p = int(np.abs(gauss)) + 1
        Np = min(max_point, gauss_p)

    if points_neg is not None:
        Nn = points_neg
    else:
        gauss = random.gauss(mu=0, sigma=max_point // 2)
        gauss_p = int(np.abs(gauss))
        Nn = min(max_point, gauss_p)

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
            npred = (previous_pred[i, 0, ...] == 0.0).float()

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
        _point_label.append(
            torch.tensor([1] * min(len(plabelpoints), Np) + [0] * min(len(nlabelpoints), Nn) + [-1] * n_placeholder).to(
                device
            )
        )

    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    point_coords = apply_coords_torch(point, max(h, w), args.sam_image_size)

    return point_coords, point_label


def prepare_sam_training_input(inputs: torch.Tensor, labels: torch.Tensor, args: Namespace, model: Vista2pt5D):
    """

    @param inputs: (B) roi_z x H x W
    @param labels: (B) H x W
    @param args:
    @param model:
    @return:
    """
    # Shape with Nc
    unique_labels: torch.Tensor | MetaTensor = torch.unique(labels)
    if hasattr(unique_labels, 'as_tensor'):
        unique_labels: torch.LongTensor = unique_labels.as_tensor().long()
    else:
        unique_labels: torch.LongTensor = unique_labels.long()

    nc_in_mask: int = len(unique_labels)
    if args.skip_bk:
        unique_labels: torch.LongTensor = unique_labels[1:]

    if nc_in_mask == 0:
        prepared_input = list()
        for batch_idx, (_inputs, _labels) in enumerate(zip(inputs, labels)):
            prepared_input.append({
                'image': _inputs,
                'original_size': tuple(_labels.shape)
            })
        # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
        batch_labels = torch.zeros(batch_idx + 1, 1, args.sam_image_size // 4, args.sam_image_size // 4)
        skip = True
        return prepared_input, batch_labels, None, skip

    # random sample args.num_prompt prompts, this will help to manage the GPU memory upper bound.
    if nc_in_mask > args.num_prompt:
        # random some category in nc_in_mask
        idxs: int = random.sample(range(nc_in_mask), args.num_prompt)
        idxs: torch.Tensor = torch.tensor(idxs)
        unique_labels: torch.LongTensor = unique_labels[idxs]

    if len(unique_labels) < args.num_prompt:
        # Cat unique_labels into unique_labels until the size of nc_in_mask(not unique now) >= num_prompt
        while len(unique_labels) < args.num_prompt:
            unique_labels: torch.LongTensor = torch.cat([unique_labels, unique_labels], 0).long()
        # make sure size of unique_labels == num_prompt
        unique_labels = unique_labels[: args.num_prompt]

    # add 4 background labels to every batch
    # The background labels is meaning
    background_labels = list(set(range(1, args.nc)) - set(unique_labels.cpu().numpy()))
    random.shuffle(background_labels)
    unique_labels: torch.LongTensor = torch.cat([unique_labels, torch.tensor(background_labels[:4]).cuda(args.rank)]).long()

    # preprocess make the size of label same as low_res_logit
    # The shape is (B, Nc, H, W)
    batch_labels_ = torch.cat([labels == unique_labels[i] for i in range(len(unique_labels))], dim=1).float()
    # The shape will become (B, NC, sam_H / 4, sam_W / 4)
    if args.distributed:
        batch_labels = model.module.preprocess(batch_labels_, is_input=False)
    else:
        batch_labels = model.preprocess(batch_labels_, is_input=False)

    # TODO: we currently only use class-label and points prompt.

    prepared_input = list()
    for batch_idx, (_inputs, _labels, _batch_labels_) in enumerate(zip(inputs, labels, batch_labels_)):
        prepared_input.append({
            'image': _inputs,
            'original_size': tuple(_labels.shape)
        })
        if args.label_prompt:
            labels_prompt = unique_labels.unsqueeze(-1)
            prepared_input[batch_idx].update({'labels': labels_prompt})
        if args.point_prompt:
            point_coords, point_labels = generate_point_prompt(_batch_labels_, args)
            prepared_input[batch_idx].update({
                'point_coords': point_coords,
                'point_labels': point_labels
            })
        if args.label_prompt and args.point_prompt:
            if random.uniform(0, 1) < args.drop_label_prob:
                prepared_input[batch_idx].pop('labels')
                continue
            if random.uniform(0, 1) < args.drop_point_prob:
                prepared_input[batch_idx].pop('point_coords')
                prepared_input[batch_idx].pop('point_labels')
    return prepared_input, batch_labels.cuda(args.rank), batch_labels_, False

    # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    # if args.label_prompt:
    #     labels_prompt = unique_labels.unsqueeze(-1)
    #     prepared_input[0].update({"labels": labels_prompt})
    #
    # if args.point_prompt:
    #     point_coords, point_labels = generate_point_prompt(batch_labels_, args)
    #     prepared_input[0].update({"point_coords": point_coords, "point_labels": point_labels})
    #
    # if args.label_prompt and args.point_prompt:
    #     # if we use both two kinds of prompts, then we randomly drop one kind.
    #     if random.uniform(0, 1) < args.drop_label_prob:
    #         prepared_input[0].pop("labels")
    #     else:
    #         if random.uniform(0, 1) < args.drop_point_prob:
    #             prepared_input[0].pop("point_coords")
    #             prepared_input[0].pop("point_labels")
    #
    # return prepared_input, batch_labels.unsqueeze(1), batch_labels_, False


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        labels_l = batch_data["label"]
        # TODO: we only support batch_size = 1 for data loader.        
        B = inputs_l.shape[0]
        n_z_before_pad = labels_l.shape[-1]

        n_slice = args.roi_z_iter
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        pd = (n_slice // 2, n_slice // 2)
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        labels_l = F.pad(labels_l, pd, "constant", 0)
        _loss = torch.tensor(0.0).cuda(args.rank)

        for _k in range(args.num_patch):
            # Return random integers from `low` (inclusive) to `high` (exclusive).
            start_idx = int(np.random.randint(low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)))

            left_ptr = start_idx - n_slice // 2
            right_ptr = start_idx + n_slice // 2 + 1
            # The shape from B (C=1) H W S -> B (C=1) H W (S=z_roi) -> B (C=1) (S=z_roi) H W
            inputs = inputs_l[..., left_ptr: right_ptr].permute(0, 1, 4, 2, 3)
            # Remove channel axis: B (C=1) (S=z_roi) H W -> B (S=z_roi) H W
            inputs = inputs.squeeze(1)

            # we only need the label for the center slice
            # B C H W S -> B C H W (S=z_roi) -> B C H W
            labels = labels_l[..., left_ptr: right_ptr][..., n_slice // 2]
            data, target, target_original, skip = prepare_sam_training_input(
                inputs.cuda(args.rank), labels.cuda(args.rank), args, model
            )

            for param in model.parameters():  # Like optimizer.zero_grad(set_to_none=True)
                param.grad = None

            with autocast(enabled=args.amp):
                outputs = model(data, is_train=True)
            # not sure this operation is correct or not, i trying to cat at channels axis(maybe)
            pred_mask = torch.cat([_out['low_res_logits'] for _out in outputs], dim=1)
            pred_mask = pred_mask.permute(1, 0, 2, 3)
            loss = loss_func(pred_mask, target)

            if skip:
                loss = loss * 0.0

            if args.amp:
                scaler.scale(loss).backward()
                if args.clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            _loss += loss.detach()
        _loss /= min(args.num_patch, n_z_before_pad)
        if args.distributed:
            loss_list = distributed_all_gather(
                [_loss],
                out_numpy=True,
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=args.num_patch)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def train_epoch_iterative(model, loader, optimizer, scaler, epoch, loss_func, run, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        labels_l = batch_data["label"]
        # TODO: we only support batch_size = 1 for data loader.
        inputs_l = inputs_l
        labels_l = labels_l
        n_z_before_pad = labels_l.shape[-1]

        n_slice = args.roi_z_iter
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        pd = (n_slice // 2, n_slice // 2)
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        labels_l = F.pad(labels_l, pd, "constant", 0)
        _loss = torch.tensor(0.0).cuda(args.rank)

        for _k in range(min(args.num_patch, n_z_before_pad)):
            # Return random integers from `low` (inclusive) to `high` (exclusive).
            start_idx = int(np.random.randint(low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)))
            left_ptr = start_idx - n_slice // 2
            right_ptr = start_idx + n_slice // 2 + 1
            # B C H W S -> B C S H W -> B S H W
            inputs = inputs_l[..., left_ptr: right_ptr].permute(0, 1, 4, 2, 3)

            # we only need the label for the center slice
            labels = labels_l[..., left_ptr: right_ptr][..., n_slice // 2]

            data, target, target_original, skip = prepare_sam_training_input(
                inputs.cuda(args.rank), labels.cuda(args.rank), args, model
            )
            for param in model.parameters():
                param.grad = None

            with autocast(enabled=args.amp):
                if args.distributed:
                    image_embeddings = model.module.get_image_embeddings(data)
                else:
                    image_embeddings = model.get_image_embeddings(data)

            if skip:
                with autocast(enabled=args.amp):
                    if args.distributed:
                        outputs = model.module.get_mask_prediction(data, image_embeddings)
                    else:
                        outputs = model.get_mask_prediction(data, image_embeddings)
                loss = loss_func(outputs[0]["low_res_logits"], target) * 0.0
            else:
                # iterative training
                loss = 0
                drop_iter = random.randint(0, args.num_iterative_step - 2)
                for i in range(args.num_iterative_step):
                    with autocast(enabled=args.amp):
                        if args.distributed:
                            outputs = model.module.get_mask_prediction(data, image_embeddings)
                        else:
                            outputs = model.get_mask_prediction(data, image_embeddings)
                    loss += loss_func(outputs[0]["low_res_logits"], target)
                    if i == args.num_iterative_step - 1:
                        # no need to perform the following operations after the last step
                        continue
                    # we also supply the mask prediction from the previous iteration
                    # as an additional prompt to our model (follow original SAM).
                    data[0]["mask_inputs"] = outputs[0]["low_res_logits"].detach()
                    if i == drop_iter:
                        # for drop iter, no additional points are sampled (follow original SAM).
                        continue

                    previous_point_coords = data[0].get("point_coords", None)
                    previous_point_labels = data[0].get("point_labels", None)

                    if previous_point_coords is None and args.no_more_points_for_cp_only:
                        # if no point prompt at the first prompt generation,
                        # we will not add more additional pointa during iterative training.
                        continue

                    # sample one pos and on neg point based on previous prediction
                    previous_pred = (F.sigmoid(outputs[0]["high_res_logits"].detach()) > 0.5).float()
                    point_coords, point_labels = generate_point_prompt(
                        target_original, args=args, points_pos=1, points_neg=1, previous_pred=previous_pred
                    )

                    if previous_point_coords is not None:
                        data[0]["point_coords"] = torch.cat([previous_point_coords, point_coords], dim=1)
                        data[0]["point_labels"] = torch.cat([previous_point_labels, point_labels], dim=1)
                    else:
                        data[0]["point_coords"] = point_coords
                        data[0]["point_labels"] = point_labels

            if args.amp:
                scaler.scale(loss).backward()
                if args.clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            _loss += loss.detach() / args.num_iterative_step
        _loss /= min(args.num_patch, n_z_before_pad)
        if args.distributed:
            loss_list = distributed_all_gather(
                [_loss],
                out_numpy=True,
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=args.num_patch)
        if args.rank == 0:
            dur = time.time() - start_time
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(dur),
            )
            if run is not None:
                run.log({
                    'train iter loss': run_loss.avg,
                    'train iter time': dur,
                })

        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def prepare_sam_test_input(inputs, labels, args, previous_pred=None):
    unique_labels = torch.tensor([i for i in range(1, args.nc)]).cuda(args.rank)

    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    if args.label_prompt:
        labels_prompt = unique_labels.unsqueeze(-1)
        prepared_input[0].update({"labels": labels_prompt})

    if args.point_prompt:
        point_coords, point_labels = generate_point_prompt(
            batch_labels,
            args,
            points_pos=args.points_val_pos,
            points_neg=args.points_val_neg,
            previous_pred=previous_pred,
        )
        prepared_input[0].update({"point_coords": point_coords, "point_labels": point_labels})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels


def prepare_sam_val_input_cp_only(inputs, labels, args):
    """

    @param inputs: A 3d tensor with shape roi_z_iter x H x W
    @param labels: A 2d tensor with shape H x W
    @param args:
    @return:
    """
    # Don't exclude background in val but will ignore it in metric calculation
    unique_labels = torch.tensor([i for i in range(1, args.nc)]).cuda(args.rank)

    """
        Some annotation for `batch_labels`
        - preprocess make the size of label same as high_res_logit.
        - As the result, just become the one-hot encoding.
        - The shape is (nc - 1, H, W). nc - 1 is for skip background
    """
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]

    labels_prompt = unique_labels.unsqueeze(-1)
    prepared_input[0].update({"labels": labels_prompt})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels


def val_epoch(model, loader, epoch, acc_func, args, iterative=False, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():

        for idx, batch_data in enumerate(loader):
            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]
            B = inputs_l.shape[0]
            n_slice = args.roi_z_iter
            # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
            pd = (n_slice // 2, n_slice // 2)

            if B == 1:
                inputs_l = inputs_l.squeeze()
                labels_l = labels_l.squeeze()

            # padding at last axis (z-axis), the goal in this step like convolution padding
            inputs_l = F.pad(inputs_l, pd, "constant", 0)
            labels_l = F.pad(labels_l, pd, "constant", 0)
            n_z_after_pad = labels_l.shape[-1]

            acc_sum_total = 0.0
            not_nans_total = 0.0
            start = n_z_after_pad // 2 - args.num_patch_val // 2
            end = n_z_after_pad // 2 + args.num_patch_val // 2
            # We only loop the center args.num_patch_val slices to save val time
            for start_idx in range(start, end):
                left_ptr = start_idx - n_slice // 2
                right_ptr = start_idx + n_slice // 2 + 1
                if B == 1:
                    inputs = inputs_l[..., left_ptr: right_ptr].permute(2, 0, 1)
                else:
                    inputs = inputs_l[..., left_ptr: right_ptr].permute(0, 1, 4, 2, 3)

                # we only need the label for the center slice
                labels = labels_l[..., left_ptr: right_ptr][..., n_slice // 2]

                data: torch.Tensor | list[torch.Tensor] = []
                target: torch.Tensor | list[torch.Tensor] = []
                data, target, _ = prepare_sam_val_input_cp_only(
                    inputs.cuda(args.rank), labels.cuda(args.rank), args
                )


                with autocast(enabled=args.amp):
                    outputs = model(data)
                    logit = torch.cat([_out['high_res_logits'] for _out in outputs], dim=0)

                y_pred = torch.stack(post_pred(decollate_batch(logit)), 0)

                # TODO: we compute metric for each prompt for simplicity in validation.
                print(y_pred.shape, target.shape)
                acc_batch = compute_dice(y_pred=y_pred, y=target)
                acc_sum, not_nans = (
                    torch.nansum(acc_batch).item(),
                    args.nc - 1 - torch.sum(torch.isnan(acc_batch).float()).item(),
                )
                acc_sum_total += acc_sum
                not_nans_total += not_nans

            acc, not_nans = acc_sum_total / not_nans_total, not_nans_total
            f_name = batch_data["image"].meta["filename_or_obj"]
            print(f"Rank: {args.rank}, Case: {f_name}, Acc: {acc:.4f}, N_prompts: {int(not_nans)} ")

            acc = torch.tensor(acc).cuda(args.rank)
            not_nans = torch.tensor(not_nans).cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx + 1, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    run = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        if args.wandb:
            print(f'Initializing wandb')
            run = wandb.init(project=args.project, name=args.name, config=args)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    best_epoch = -1
    val_MA = None
    best_log = {}
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        if args.rank == 0:
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = optimizer.param_groups[0]["lr"]
            print("Current lr:", lr)
            if run is not None:
                run.log({
                    'lr': lr,
                    'epoch': epoch
                })

        if args.label_prompt and args.point_prompt:
            if epoch < args.label_prompt_warm_up_epoch:
                # during warm up, we drop class label prompt embedding with less prob,
                # since class label prompt embedding layer is trained from scratch.
                args.drop_label_prob = 0.2
                args.drop_point_prob = 0.5
            else:
                # after warmp up, we evenly drop two kinds of prompts
                args.drop_label_prob = 0.5
                args.drop_point_prob = 0.5
            print(
                "rank:",
                args.rank,
                "label_prompt (train):",
                args.label_prompt,
                ", label_drop_prob:",
                args.drop_label_prob,
                "| point_prompt (train):",
                args.point_prompt,
                ", point_drop_prob:",
                args.drop_point_prob,
            )

        # we don't perform iterative training for the first args.iterative_training_warm_up_epoch epochs
        if epoch > args.iterative_training_warm_up_epoch:
            if args.reuse_img_embedding:
                if args.rank == 0:
                    print("Iterative Training: Reuse image embedding!")
                train_loss = train_epoch_iterative(
                    model, train_loader, optimizer,
                    scaler=scaler, epoch=epoch, loss_func=loss_func,
                    run=run, args=args
                )
            else:
                if args.rank == 0:
                    print("Iterative Training: Don't reuse image embedding!")
                raise NotImplementedError
        else:
            print(f" Rank: {args.rank} Single-step Training")
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
            )

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            if writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)
            if run is not None:
                run.log({
                    'train_loss': train_loss,
                    'epoch': epoch
                })

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            if args.rank == 0:
                print("Start validation")
                print("label_prompt (val):", args.label_prompt, "point_prompt (val):", args.point_prompt)
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                iterative=False,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)
            if val_MA is None:
                val_MA = val_avg_acc
            else:
                val_MA = 0.9 * val_MA + 0.1 * val_avg_acc
            if args.rank == 0:
                print(
                    "Final validation  {}/{},".format(epoch, args.max_epochs - 1),
                    f"Acc {val_avg_acc:.4f},",
                    f"mv Acc {val_MA:.4f},",
                    "Previous Best validation at epoch {} is {:.4f},".format(best_epoch, val_acc_max),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if run is not None:
                    run.log({
                        'val_acc': val_avg_acc,
                        'epoch': epoch
                    })

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    best_log[epoch] = float(val_acc_max)
                    best_epoch = epoch
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model,
                            epoch,
                            args,
                            best_acc=val_acc_max,
                            filename="model_best.pt",
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                with open(os.path.join(args.logdir, "train.log"), "w") as f:
                    json.dump(best_log, f)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0 and writer is not None:
        writer.close()

    print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch)

    return val_acc_max


if __name__ == '__main__':
    cargs = Namespace(
        skip_bk=False, sam_image_size=128, num_prompt=37, nc=11, distributed=False, label_prompt=True,
        point_prompt=True, drop_label_prob=.5, drop_point_prob=.5, max_points=11
    )
    model = sam_model_registry['vit_b'](
        image_size=cargs.sam_image_size,
        encoder_in_chans=81,
        patch_embed_3d=True
    )
    B = 1
    images = torch.randn((B, 1, cargs.sam_image_size, cargs.sam_image_size, 27))
    LABELS = torch.randint(0, 11, (B, cargs.sam_image_size, cargs.sam_image_size))
    prepare_data, target, target_org, boolean = prepare_sam_training_input(images.squeeze(), LABELS.squeeze(), cargs, model)
    # print(prepare_data['image'].shape, prepare_data['labels'].shape)
    # print(target.shape)
