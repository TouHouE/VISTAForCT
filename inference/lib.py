import torch
from monai import transforms as MF
from monai.data.utils import decollate_batch
import os
from torch.nn import functional as F
from inference import utils as IU

class Processor:
    def __init__(self, args):
        self.args = args
        self.loader = MF.LoadImage()
        self.channeler = MF.EnsureChannelFirst()
        self.axior = MF.Orientation(axcodes="RAS")
        self.spacier = MF.Spacing((1.5, 1.5, 1.5))
        self.resizer = MF.Resize((args.image_size, args.image_size, 320))
        self.scaler = MF.ScaleIntensityRange(
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max,
            clip=args.clip
        )
        self.pad_size = (args.roi_z_iter // 2, args.roi_z_iter // 2)
        self.output_method = MF.Compose([MF.Activations(sigmoid=True), MF.AsDiscrete(threshold=.5)])

    def prepare_input(self, image):
        vista_input, uni_labels = IU.prepare_slice_data(image, self.args)
        return vista_input, uni_labels

    def prepare_output(self, output_cand: list[torch.Tensor]):
        """

        @param output_cand: a list of prediction mask with shape: S x (nc, 1, H, W), the nc is # of category
        @return: A 3d mask with shape: (nc, S, H, W)
        """
        # print(output_cand.shape)
        # output_cand = decollate_batch(output_cand)
        output_cand = self.output_method(output_cand)
        return torch.cat(output_cand, 1)

    def __call__(self, path, need_affine=False, is_label=False) -> torch.Tensor:
        if (img_folder := self.args.image_folder) is not None:
            path = os.path.join(img_folder, path)
        image = self.loader(path)
        if need_affine:
            am = image.affine
        image = self.channeler(image)
        image = self.axior(image)
        # image = self.spacier(image)
        # image = self.resizer(image)
        if not is_label:
            image = self.scaler(image)
        # image = F.pad(image, self.pad_size, 'constant', 0)
        if need_affine:
            return image, am
        return image
