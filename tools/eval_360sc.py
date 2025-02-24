from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import numpy as np

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup

palette = [128, 64, 128, 
244, 35, 232, 
70, 70, 70, 
102, 102, 156, 
190, 153, 153, 
153, 153, 153, 
250, 170, 30,
220, 220, 0, 
107, 142, 35, 
152, 251, 152, 
70, 130, 180, 
220, 20, 60, 
# 255, 0, 0, 
0, 0, 142, 
# 0, 0, 70,
# 0, 60, 100, 
# 0, 80, 100, 
# 0, 0, 230, 
# 119, 11, 32
]
zero_pad = 256 * 3 - len(palette)


NAME_CLASSES = ['terrain', 'sky', 'thin cloud', 'thick cloud']

for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8))
    new_mask.putpalette(palette)

    return new_mask


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])  

        # dataset and dataloader
        # crop_size = cfg.TEST.CROP_SIZE
        val_dataset = get_segmentation_dataset('skycloud360', split='test', mode='testval', transform=input_transform)
        # val_dataset = get_segmentation_dataset('stanford2d3d_pan', split='trainval', mode='val', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model.evaluate(image)

            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            save_vis = True
            if save_vis:
                output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
                output_col = colorize_mask(output)
                output = Image.fromarray(output.astype(np.uint8))
                name = filename[0].split('/')[-1]
                img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * cfg.DATASET.STD + cfg.DATASET.MEAN) * 255
                image = Image.fromarray(img_np.astype(np.uint8))
                image.save('%s/%s_image.png' % ('output', name.split('.')[0]))
                output.save('%s/%s' % ('output', name))
                output_col.save('%s/%s_color.png' % ('output', name.split('.')[0]))
                print('Saved to %s/%s' % ('output', name))
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    # cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
