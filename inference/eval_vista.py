import os
from argparse import ArgumentParser
from typing import Optional
from collections import OrderedDict
import torch
from torch import nn
from training.segment_anything import sam_model_registry
from monai import transforms as MF
from inference.lib import Processor
import json
from tqdm.auto import tqdm

def make_data_pack_list(args):
    json_file = getattr(args, 'json_form', None)
    file_path = getattr(args, 'file_path', None)
    pack_list = list()

    if json_file is not None:
        with open(json_file, 'r') as jin:
            _pack_list = json.load(jin)
        pack_list.extend(_pack_list['testing'])
    elif json_file is None and file_path is not None:
        pack_list.append({
            'image': file_path
        })
    return pack_list


def load_model(ckpt_path, model: nn.Module) -> nn.Module:
    old_state_dict = torch.load(ckpt_path, map_location='cpu')
    weight_dict = OrderedDict()

    if (_mapper := old_state_dict.get('state_dict')) is None:
        for k, v in _mapper.items():
            weight_dict[k] = v
    else:
        weight_dict = old_state_dict.copy()
    return model.load_state_dict(weight_dict)


@torch.no_grad()
def launch_eval(model: nn.Module, data_pack_list: list, processor: Processor, args) -> list[torch.Tensor]:
    def _one_slice(_model: nn.Module, _slice_image) -> torch.Tensor:
        _slice_pred = _model(_slice_image)
        return _slice_pred[0]['high_res_logits']
    model.eval()
    model.cuda()

    for idx, dpack in tqdm(enumerate(data_pack_list), total=len(data_pack_list)):
        image_path = dpack['image']
        image = processor(image_path)
        image = image.cuda()
        # C, H, W, S -> H, W, S -> H, W, Ns, S' -> Ns, S', H, W
        slice_group = image[0].unfold(-1, args.roi_z_iter, 1).permute(2, 3, 0, 1)
        pred_group = list()
        for slice_image in slice_group:
            vista_input, labels_name = processor.prepare_input(slice_image)

            slice_mask_pred = _one_slice(model, vista_input)
            pred_group.append(processor.prepare_output(slice_mask_pred))
        torch.save(torch.stack(pred_group, 0), os.path.join(args.output_folder, f'pred_{image_path}.pt'))


def main(args):
    model = sam_model_registry[args.vit_type](
        image_size=args.image_size,
        encoder_in_chans=args.roi_z_iter * 3,
        patch_embed_3d=args.patch_embed_3d
    )
    model = load_model(args.ckpt_path, model)
    processor = Processor(args)
    data_pack_list = make_data_pack_list(args)
    launch_eval(model, data_pack_list, processor)



def make_sure_folder_exist(args):
    os.makedirs(args.output_folder, exist_ok=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path')
    parser.add_argument('--file_path')
    parser.add_argument('--json_form')
    parser.add_argument('--image_folder', default='/mnt/src/data')
    parser.add_argument('--output_folder', default='./out')
    parser.add_argument('--vit_type', default='vit_h')
    parser.add_argument('--image_size', default=512)
    parser.add_argument('--roi_z_iter', default=27)
    parser.add_argument('--patch_embed_3d', action='store_true', default=False)
    parser.add_argument('--a_min', default=-1024)
    parser.add_argument('--a_max', default=1024)
    parser.add_argument('--b_min', default=-1)
    parser.add_argument('--b_max', default=1)
    parser.add_argument('--clip', action='store_true', default=True)
    args = parser.parse_args()
    make_sure_folder_exist(args)
    main(args)