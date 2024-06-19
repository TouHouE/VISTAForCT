import os
import json
from collections import OrderedDict
from argparse import ArgumentParser
from typing import Optional, Callable

import wandb

import torch
from torch import nn
from tqdm.auto import tqdm

from monai import transforms as MF
from monai.data import MetaTensor
from monai.metrics import compute_dice

import numpy as np

from training.vista_2pt5d.model import sam_model_registry

from inference import other
from inference.lib import Processor
labels = [
        "Background",
        'RightAtrium',
        'RightVentricle',
        'LeftAtrium',
        'LeftVentricle',
        'MyocardiumLV',
        'Aorta',
        'Coronaries8',
        'Fat',
        'Bypass',
        'Plaque']

def make_data_pack_list(args):
    json_file = getattr(args, 'json_form', None)
    file_path = getattr(args, 'file_path', None)
    pack_list = list()

    if json_file is not None:
        with open(json_file, 'r') as jin:
            _pack_list = json.load(jin)
        pack_list.extend(_pack_list['training'])
    elif json_file is None and file_path is not None:
        pack_list.append({
            'image': file_path
        })
    return pack_list


def load_model(ckpt_path, model: nn.Module) -> nn.Module:
    old_state_dict = torch.load(ckpt_path, map_location='cpu')
    weight_dict = OrderedDict()
    print(old_state_dict.keys())
    if (_mapper := old_state_dict.get('state_dict')) is not None:
        for k, v in _mapper.items():
            weight_dict[k] = v
    else:
        weight_dict = old_state_dict.copy()
    model.load_state_dict(weight_dict)
    return model


@torch.no_grad()
def launch_eval(model: nn.Module, data_pack_list: list, processor: Processor, args) -> list[torch.Tensor]:
    global labels
    def _one_slice(_model: nn.Module, _slice_image) -> torch.Tensor:
        _slice_pred = _model(_slice_image)
        return _slice_pred[0]['high_res_logits']
    def my_method():

        # C, H, W, S -> H, W, S -> H, W, Ns, S' -> Ns, S', H, W
        slice_group = image[0].unfold(-1, args.roi_z_iter, 1).permute(2, 3, 0, 1)
        pred_group = list()
        bar2 = tqdm(slice_group, total=len(slice_group), leave=True)
        for slice_image in bar2:
            vista_input, labels_name = processor.prepare_input(slice_image)

            slice_mask_pred = _one_slice(model, vista_input)
            pred_group.append(slice_mask_pred)
        return processor.prepare_output(pred_group)

    model_date: str = args.ckpt_path.split('/')[-2]
    model_type: str = args.ckpt_path.split('/')[-3]
    model.eval()
    model.cuda()
    table: dict = dict()
    saver: Callable = MF.SaveImage(args.output_folder, output_postfix='pred', output_dtype=torch.int16)
    print(f'Config: {args}')
    
    if args.wandb:
        wandb.init(project='show_seg', name=f'{model_type}_{model_date}')
    label_map: dict[int, str] = {idx: key for idx, key in enumerate(labels)}
    best_dice = -1
    best_image_obj = None

    for idx, dpack in tqdm(enumerate(data_pack_list), total=len(data_pack_list)):
        image_path: str = dpack['image']
        image_name: str = image_path.split('/')[-1]
        label_path: str = dpack['label']
        image, affine = processor(image_path, True)
        label: MetaTensor = processor(label_path, is_label=True)
        image: MetaTensor = image.cuda().unsqueeze(0)

        mask3d: MetaTensor = other.vista_slice_inference(
            image, model, 'cuda', n_z_slices=args.roi_z_iter,
            labels=labels, computeEmbedding=False,
            class_prompts=args.class_prompts, point_prompts=args.point_prompts,
            cached_data=False, cachedEmbedding=False,
            original_affine=affine
        )
        fully_dice = .0

        for s in range(image.shape[-1]):
            # print(mask3d.shape)
            slice_mask: np.ndarray = mask3d[0, 0, ..., s].detach().cpu().numpy()
            slice_label: np.ndarray = label[0, ..., s].detach().cpu().numpy()
            dice_group = compute_dice(y_pred=mask3d[0, ..., s].cpu(), y=label[..., s].cpu())
            total_dice = torch.nansum(dice_group).item()
            num_of_not_nan = args.nc - 1 - torch.sum(torch.isnan(dice_group).float()).item()
            current_dice = total_dice / num_of_not_nan
            fully_dice += current_dice

            mask_pack = {
                    'predictions': {
                        'mask_data': slice_mask,
                        'class_labels': label_map,
                    }, 
                    'ground_truth': {
                        'mask_data': slice_label,
                        'class_labels': label_map
                    }
                }
            image_obj = wandb.Image(
                    image[0, 0, ..., s].detach().cpu().numpy(),
                    masks=mask_pack,
                    caption=f'slice:{s}-Dice: {current_dice:.5f}'
                    )
            if args.wandb:    
                wandb.log({image_name: image_obj, 'slice': s, 'path': image_path, 'dice score': current_dice})
            if current_dice > best_dice:
                best_dice = current_dice
                best_image_obj = image_obj
        if args.wandb:
            wandb.log({'best for each image': best_image_obj})


        # print(f'final shape: {mask3d.shape}')
        saver(mask3d.squeeze(0), mask3d.meta)
        # torch.save(mask3d, )
        table[f'pred_{idx}'] = image_path
        if args.debug > -1 and idx >= args.debug:
            break
    # for loop end
    with open(os.path.join(args.output_folder, 'table.json'), 'w+') as jout:
        json.dump(table, jout)



def main(args):

    model = sam_model_registry[args.vit_type](
        image_size=args.image_size,
        encoder_in_chans=args.roi_z_iter * 3,
        patch_embed_3d=args.patch_embed_3d
    )
    model = load_model(args.ckpt_path, model)
    processor = Processor(args)
    data_pack_list = make_data_pack_list(args)
    launch_eval(model, data_pack_list, processor, args)



def make_sure_folder_exist(args):
    os.makedirs(args.output_folder, exist_ok=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path')
    parser.add_argument('--file_path')
    parser.add_argument('--json_form')
    parser.add_argument('--class_prompts', action='store_true', default=False)
    parser.add_argument('--point_prompts', action='store_true', default=False)
    parser.add_argument('--num_prompt', default=0, type=int, required=False)
    parser.add_argument('--image_folder', default='/mnt/src/data')
    parser.add_argument('--output_folder', default='./out')
    parser.add_argument('--nc', default=11, type=int, help='including background(0).')
    parser.add_argument('--vit_type', default='vit_h')
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--roi_z_iter', default=27, type=int)
    parser.add_argument('--patch_embed_3d', action='store_true', default=False)
    parser.add_argument('--a_min', default=-1024, type=int)
    parser.add_argument('--a_max', default=1024, type=int)
    parser.add_argument('--b_min', default=-1, type=int)
    parser.add_argument('--b_max', default=1, type=int)
    parser.add_argument('--clip', action='store_true', default=True)
    parser.add_argument('--debug', default=-1, type=int, required=False, help='greater than 0 into debug mode')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--project_name', default='show_seg', required=False, help='using to wandb project')
    args = parser.parse_args()
    make_sure_folder_exist(args)
    main(args)
