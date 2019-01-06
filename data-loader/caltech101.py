# -*- coding: utf-8 -*-
"""Caltech-101 dataset for PyTorch loading.

Caltech-101 dataset (Li et al., 2007) contains from 31 to 800 images per
category. Most images are medium resolution, i.e., about 300*300 pixels.
Caltech-101 is diverse, though it is not without shortcomings. Namely, most
images feature relatively little clutter, and the objects are centered and
occupy most of the image. In addition, a number of categories, such as minaret,
are affected by "corner" artifacts resulting from artificial image rotation.
Though these artifacts are semantically irrelevant, they can provide stable
cues resulting in misleadingly high recognition rates.

We follow the experimental setup of (Lazebnik et al., 2006), namely, we train
on 30 images per class and test on the rest. For efficiency, we limit the
number of test images to 50 per class. Note that, because some categories are
very small, we may end up with just a single test image per class. The
BACKGROUND_Google class is not used in neither training or testing. As a
result, training dataset size is 3030, while testing dataset size is 2945.
Experiments are conducted five time (with different random seed).

Data preparation
    Download dataset from:
    http://www.vision.caltech.edu/Image_Datasets/Caltech101/

(Li et al., 2007) F.-F. Li, R. Fergus, P. Perona. Learning generative visual
    models from few training examples: An incremental Bayesian approach tested
    on 101 object categories. Computer Vision and Image Understanding,
    106(1): 59-70, 2007.
"""


import collections
import os
import random

import PIL.Image
import numpy as np
import torch


__all__ = ['Caltech101']
__author__ = 'Hao Zhang'
__copyright__ = '2019 LAMDA'
__date__ = '2019-01-04'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2019-01-04'
__version__ = '1.0'


class Caltech101(torch.utils.data.Dataset):
    """Caltech-101 dataset.

    Attributes:
        _root, str: Dataset path.
        _is_train, bool: Train/Validation.
        _transform, torchvision.transforms: Data augmentations.
        _seed, int: Random seed for train/val split.
        _train_images, list<np.ndarray>.
        _train_labels, list<int>.
        _val_images, list<np.ndarray>.
        _val_labels, list<int>.
    """
    def __init__(self, root, is_train, transform, seed):
        self._root = os.path.expanduser(root)
        self._is_train = is_train
        self._transform = transform
        self._seed = seed

        if not os.path.isfile(os.path.join(self._root, 'all_data.pth')):
            self._prepareDataset()

        train_path = os.path.join(root, 'train_seed_%d.pth' % self._seed)
        val_path = os.path.join(root, 'val_seed_%d.pth' % self._seed)
        if (not os.path.isfile(train_path) or not os.path.isfile(val_path)):
            self._splitTrainVal()

        if self._is_train:
            self._train_images, self._train_labels = torch.load(train_path)
        else:
            self._val_images, self._val_labels = torch.load(val_path)

    def _prepareDataset(self):
        """Load all images into np.ndarray and get all labels.

        The results are stored onto disk as all_data.pth.
        """
        print('Prepare Caltech-101 dataset.')
        raw_image_path = os.path.join(self._root, '101_ObjectCategories')
        assert os.path.isdir(raw_image_path)

        all_class_names = [file_.name for file_ in os.scandir(raw_image_path)
                           if (file_.is_dir()
                               and file_.name != 'BACKGROUND_Google')]
        assert len(all_class_names) == 101
        all_class_names.sort()

        label_to_image = collections.OrderedDict()
        for label_index in range(len(all_class_names)):
            label_to_image[label_index] = []
        for label_index, class_name in enumerate(all_class_names):
            class_path = os.path.join(raw_image_path, class_name)
            with os.scandir(class_path) as it:
                for file_ in it:
                    if file_.is_file():
                        image = PIL.Image.open(os.path.join(
                            class_path, file_.name))
                        if image.mode == 'L':
                            image = image.convert('RGB')
                        image = np.asarray(image)
                        label_to_image[label_index].append(image)
        torch.save((label_to_image, all_class_names),
                   os.path.join(self._root, 'all_data.pth'))

    def _splitTrainVal(self):
        """Split train/val data. More details can be referred in the README
        file.

        The results are stored onto disk as train_seed_%d.pth and
        val_seed_%d.pth.
        """
        print('Split train/val for Caltech-101 with seed=%d.' % self._seed)
        label_to_image, all_class_names = torch.load(os.path.join(
            self._root, 'all_data.pth'))
        random.seed(self._seed)
        train_images, train_labels = [], []
        val_images, val_labels = [], []
        for label_index, images in label_to_image.items():
            random.shuffle(images)
            sampled_train = images[:30]
            sampled_val = images[30:80]
            train_images.extend(sampled_train)
            val_images.extend(sampled_val)
            train_len = len(sampled_train)
            val_len = len(sampled_val)
            train_labels.extend([label_index for _ in range(train_len)])
            val_labels.extend([label_index for _ in range(val_len)])
            print('Class %s: train=%d, val=%d' % (
                all_class_names[label_index], train_len, val_len))
            assert val_len >= 1
        assert len(train_images) == len(train_labels)
        assert len(val_images) == len(val_labels)

        torch.save((train_images, train_labels),
                   os.path.join(self._root, 'train_seed_%d.pth' % self._seed))
        torch.save((val_images, val_labels),
                   os.path.join(self._root, 'val_seed_%d.pth' % self._seed))

    def __getitem__(self, index):
        """Integer indexing in range [0, len(self)).

        Args:
            index, int.

        Returns:
            image, torch.Tensor.
            label, int.
        """
        image = (self._train_images[index] if self._is_train
                 else self._val_images[index])
        label = (self._train_labels[index] if self._is_train
                 else self._val_labels[index])

        image = PIL.Image.fromarray(image.astype(np.uint8))
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def __len__(self):
        """Size of the dataset.

        Returns:
            length, int.
        """
        return (len(self._train_images) if self._is_train
                else len(self._val_images))
