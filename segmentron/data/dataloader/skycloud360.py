"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset
import random

class SkyCloud360Segmentation(SegmentationDataset):
    """SkyCloud360 Semantic Segmentation Dataset."""
    NUM_CLASS = 5

    def __init__(self, root='datasets/SkyCloud360', split='train', mode=None, transform=None, **kwargs):
        super(SkyCloud360Segmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), f'Please put dataset in {root}'
        self.images, self.mask_paths = _get_sky_pairs(self.root, self.split)
        # self.crop_size = [1664, 832]  # for inference only
        print(f'Found {len(self.images)} images in the folder {self.root}')
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of: {root} \n')
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                            #   23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, 1, 2, 3, 4])
        # self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

        # return torch.LongTensor(np.array(mask).astype('int32'))

    def _map(self, mask):

        values = np.unique(mask)
        new_mask = np.zeros_like(mask)
        new_mask -= 1
        for value in values:
            if value == 255: 
                new_mask[mask==value] = -1
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _val_sync_transform_resize(self, img, mask):
        w, h = img.size
        # x1 = random.randint(0, w - self.crop_size[1])
        # y1 = random.randint(0, h - self.crop_size[0])
        # img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        # mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        short_size = self.crop_size
        img = img.resize(short_size, Image.BICUBIC)
        mask = mask.resize(short_size, Image.NEAREST)
        # logging.info('img.size:', img.size)
        # logging.info('mask.size:', mask.size)        
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index]).convert('L')
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        # target = self._class_to_index(np.array(mask).astype('int32'))
        mask = self._map(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('unlabeled', 'terrain', 'sky', 'thick cloud', 'thin cloud')


def _get_sky_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):

            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    maskname = filename.replace('images', 'masks')
                    maskname = maskname.replace('.png', '_mask.png')
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logging.info('cannot find the mask or image:', imgpath, maskpath)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'images/')
        mask_folder = os.path.join(folder, 'masks/')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'test'
        val_img_folder = os.path.join(folder, 'images/')
        val_mask_folder = os.path.join(folder, 'masks/')
        img_paths, mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
      
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = SkyCloud360Segmentation()
