import argparse
import logging
import sys
import os
from attr import attr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import torch
import torch.utils
from math import ceil
from segmentron.dataloader.seg_data_base import  SegmentationDataset
# from data import BaseDataset, StatsDataset
# from config import cfg

log = logging.getLogger(__name__)
# global plt settings
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = 'dashed'
plt.rcParams['figure.figsize'] = 30, 11.5
plt.rcParams.update({'font.size': 42})

target_attributes = ['daylight', 'sunrise/sunset', 'dawn/dusk', 'night', 'spring', 'summer', 'autumn', 'winter', 'sunny', 'snow', 'rain', 'fog']

threshold = 0.75

def stats(num_classes, num_images, num_attributes):
    class_image_count = np.zeros(num_classes)
    attr_image_count = np.zeros(num_attributes)

    attr_image_percentage = np.zeros(num_attributes)
    class_pixel_count = np.zeros(num_classes)
    class_pixel_percentage = np.zeros(num_classes)
    class_percentage_per_image = np.zeros(num_classes)
    class_heatmap = np.zeros((num_classes, 100, 100))

    log.info('Number of images: {0}'.format(num_images))
    log.info('Number of classes: {0}'.format(num_classes))

    for i, (lbl, lbl_, attr) in tqdm.tqdm(enumerate(torch_data_loader), total=len(data_loader), ascii=True):
        if  not lbl.nelement() == 0:
            labels = np.int64(np.squeeze(lbl.numpy()/255))
            labels_ = np.squeeze(lbl_.numpy())/255.0
            pixel_count_ = np.bincount(labels.flatten(), minlength=num_classes)
            for c in range(num_classes):
                if pixel_count_[c] > 0:
                    class_image_count[c] += 1.0
                    class_percentage_per_image[c] += pixel_count_[c] / labels.size
                    class_pixel_count[c] += pixel_count_[c]
                    class_heatmap[c][labels_ == c] += 1.0
        
        if not attr.nelement() == 0:
            for i in range(num_attributes):
                attributes = np.squeeze(attr.numpy())
                # log.info(attributes)
                if attributes[i] > threshold:
                    attr_image_count[i] += 1              

        log.debug(class_pixel_count)
        log.debug(class_image_count)

    for c in range(num_attributes):
        attr_image_percentage[c] = attr_image_count[c] * 100.0 / num_images


    for c in range(num_classes):
        class_pixel_percentage[c] = class_pixel_count[c] * 100.0 / np.sum(class_pixel_count)
        if class_image_count[c]:
            class_percentage_per_image[c] = class_percentage_per_image[c] * 100.0 / class_image_count[c]
        # Normalize heatmap
        if class_heatmap[c].max():
            class_heatmap[c] /= np.max(class_heatmap[c])

    log.info('Sum of percentages of pixels: {0}'.format(np.sum(class_pixel_percentage)))
    log.info("Summary:")
    for c in range(num_attributes):
        log.info('Class {} appears in {:d} images, {:.2f}%'.format(
            c,
            int(attr_image_count[c]),
            attr_image_percentage[c]))
    np.savez(
        os.path.join(args.out, 'results'),
        attr_image_count = attr_image_count,
        attr_image_percentage=attr_image_percentage,
        class_pixel_percentage=class_pixel_percentage,
        class_percentage_per_image=class_percentage_per_image,
        class_heatmap=class_heatmap)
    return attr_image_count, attr_image_percentage, class_pixel_percentage, class_percentage_per_image, class_heatmap
    

def plot_image_percentage(num_attributes, attr_image_percentage):
    # Plot per-class image percentage
    x_labels_ = range(0, num_attributes)
    x_values_ = np.arange(len(x_labels_))
    y_values_ = attr_image_percentage
    y_label_ = 'Percentage of Images (%)'
    fig, ax = plt.subplots()
    barlist = ax.barh(x_values_, y_values_)
    # for c in range(num_attributes):
    #     barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    ax.set_xlim(0, 20)
    ax.set_xlabel(y_label_)
    ax.set_xticks(np.round(np.linspace(0, 20, 5), 2))
    ax.set_ylabel('Class')
    ax.set_yticks(x_values_)
    ax.set_yticklabels(target_attributes)
    # ax.set_xlim(0, 5 * ceil(attr_image_percentage.max() / 5.0))
    ax.invert_yaxis()
    ax.set_title('Class coverage')
    ax.set_xlabel('Average class presence [%]')
    ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['right'].set_linestyle('dashed')
    ax.spines['right'].set_linewidth(.2)  
    ax.spines['bottom'].set_visible(False)
    # plt.gca().margins(y=0.01)
    # plt.gcf().set_size_inches(9, 0.17 * num_attributes)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, 'class_dist.png'))

    # # Plot per-class pixel percentage
    # x_labels_ = range(0, num_classes)
    # x_values_ = np.arange(len(x_labels_))
    # y_values_ = class_pixel_percentage
    # y_label_ = 'Percentage of Pixels (%)'
    # fig, ax = plt.subplots()
    # barlist = ax.barh(x_values_, y_values_)
    # for c in range(num_classes):
    #     barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    # ax.set_yticks(x_values_)
    # ax.set_yticklabels(x_labels_)
    # ax.invert_yaxis()
    # # ax.set_xlim(0, class_pixel_percentage.max())
    # ax.set_xlabel(y_label_)
    # ax.set_ylabel('Class #')
    # ax.set_xlim(0, 5 * ceil(class_pixel_percentage.max() / 5.0))
    # ax.set_title('Pixel Percentage')
    # plt.gca().margins(y=0.01)
    # plt.gcf().set_size_inches(
    #     plt.gcf().get_size_inches()[0], 0.17 * num_classes)
    # fig.savefig(os.path.join(args.out, 'pixel_dist.png'))

    # # Plot per-class pixel percentage
    # x_labels_ = range(0, num_classes)
    # x_values_ = np.arange(len(x_labels_))
    # y_values_ = class_percentage_per_image
    # y_label_ = 'Percentage of Pixels (%)'
    # fig, ax = plt.subplots()
    # barlist = ax.barh(x_values_, y_values_)
    # for c in range(num_classes):
    #     barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    # ax.set_yticks(x_values_)
    # ax.set_yticklabels(x_labels_)
    # ax.invert_yaxis()
    # # ax.set_xlim(0, class_percentage.max())
    # ax.set_xlabel(y_label_)
    # ax.set_ylabel('Class #')
    # ax.set_title('Average class coverage')
    # ax.set_xlim(0, 5 * ceil(class_percentage_per_image.max() / 5.0))
    # plt.gca().margins(y=0.01)
    # plt.gcf().set_size_inches(
    #     plt.gcf().get_size_inches()[0], 0.17 * num_classes)
    # fig.savefig(os.path.join(args.out, 'image_pixel_dist.png'))

    # Plot heatmaps
def plot_heatmaps(num_classes, class_heatmap, class_names):
    for c in range(num_classes):
        fig, ax = plt.subplots()
        # red-blue palette
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(class_heatmap[c], ax=ax, vmin=0.0, vmax=1.0)
        ax.set_title('{0} heatmap'.format(class_names[c]))
        plt.axis('off')
        fig.savefig(os.path.join(args.out, 'heatmap_{0}.png'.format(c)))
        plt.close()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser_ = argparse.ArgumentParser(description='Calculate per-class statistics from configuration file')
    parser_.add_argument('--cfg', nargs='?', type=str, default='',
                         help='Configuration file to calculate statistics for')
    parser_.add_argument('--out', nargs='?', type=str, default='plots',
                         help='Name of output folder')
    args = parser_.parse_args()

    cfg.merge_from_file(args.cfg)
    data_loader = StatsDataset(
        cfg.DATASET.root_dataset, cfg.DATASET.list_stats, cfg.DATASET)

    class_colors = [[0, 0, 0], [128, 128, 128]]
    class_names = ['not sky', 'sky']
    log.info('Loading data from: {}'.format(cfg.DATASET.list_stats))
    torch_data_loader = torch.utils.data.DataLoader(data_loader, batch_size=1, num_workers=16, shuffle=False)

    num_classes = cfg.DATASET.num_seg_class
    num_attributes = sum(list(cfg.DATASET.num_attr_class))
    num_images = len(data_loader)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # attr_image_count, attr_image_percentage, _, _, _ = stats(num_classes, num_images, num_attributes)
    data = np.load('/home/chge7185/repositories/outdoor_attribute_estimation/plots/all_images/results.npz')
    attr_image_count = data['attr_image_count']
    attr_image_percentage = data['attr_image_percentage']
    plot_image_percentage(num_attributes, attr_image_percentage)