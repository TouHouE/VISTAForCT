import numpy as np
import nibabel as nib
import json


def make_dim():
    _xy, _z = np.random.choice([.75, 3, 2.75, 1.5, 2.25, 2.5, 1.12], size=2)
    return _xy, _xy, _z, 1


def make_bias():
    return 30 * np.random.random((1, 3))

table = dict(training=[], testing=[])
fold_list: np.ndarray = np.random.choice([0, 1], size=20, replace=True)
fold_list[-4:] = -1
print(fold_list)
for i, cur_fold in zip(range(20), fold_list):

    if i > 15 and False:
        random_shape = (512, 512, np.random.randint(30, 641))
        random_dim = make_dim()
        bias = make_bias()
        affine = np.zeros((4, 4))
        np.fill_diagonal(affine, random_dim)
        affine[:3, -1] = bias

        image = np.random.randn(*random_shape)
        mask = np.random.randint(0, 11, random_shape)
        image = nib.Nifti1Image(image, affine=affine)
        mask = nib.Nifti1Image(mask, affine=affine)
        nib.save(image, rf'D:\Data\pseudo_sample\image_{i}.nii.gz')
        nib.save(mask, rf'D:\Data\pseudo_sample\mask_{i}.nii.gz')
    print(cur_fold)
    if cur_fold < 0:
        table['testing'].append({
            'image': f'image_{i}.nii.gz',
            'label': f'mask_{i}.nii.gz'
        })
        continue

    table['training'].append({
        'image': f'image_{i}.nii.gz',
        'label': f'mask_{i}.nii.gz',
        'fold': cur_fold.tolist()
    })
with open(r'D:\Data\pseudo_sample\table.json', 'w+') as jout:
    json.dump(table, jout)