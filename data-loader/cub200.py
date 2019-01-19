# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle

import numpy as np
import PIL.Image
import torch
import torch.utils.data


torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


__all__ = ['CUB200', 'CUB200ReLU43']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2019-01-11'
__version__ = '2.0'


class CUB200(torch.utils.data.Dataset):
    """CUB200 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_instances, list<np.ndarray>.
        _train_labels, list<int>.
        _test_instances, list<np.ndarray>.
        _test_labels, list<int>.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform
        self.name = 'cub200'

        if (os.path.isfile(os.path.join(self._root, 'train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'test.pkl'))):
            print('Dataset already downloaded and verified.')
        elif download:
            url = ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
                   'CUB_200_2011.tgz')
            self._download(url)
            self._extract()
        else:
            self._extract()

        # Now load the picked data.
        if self._train:
            self._train_instances, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'train.pkl'), 'rb'))
            assert (len(self._train_instances) == 5994
                    and len(self._train_labels) == 5994)
        else:
            self._test_instances, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'test.pkl'), 'rb'))
            assert (len(self._test_instances) == 5794
                    and len(self._test_labels) == 5794)

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0o775)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw', 'CUB_200_2011.tgz')
        try:
            print('Downloading ' + url + ' to ' + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, 'r:gz')
        os.chdir(os.path.join(self._root, 'raw'))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'raw', 'CUB_200_2011', 'images')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(
            self._root, 'raw', 'CUB_200_2011', 'images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'raw', 'CUB_200_2011', 'train_test_split.txt'),
            dtype=int)

        train_instances = []
        train_labels = []
        test_instances = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_instances.append(image_np)
                train_labels.append(label)
            else:
                test_instances.append(image_np)
                test_labels.append(label)

        torch.save((train_instances, train_labels),
                   os.path.join(self._root, 'processed', 'train.pth'))
        torch.save((test_instances, test_labels),
                   os.path.join(self._root, 'processed', 'test.pth'))

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            instance, PIL.Image: Image of the given index.
            label, int: target of the given index.
        """
        if self._train:
            instance, label = (self._train_instances[index],
                               self._train_labels[index])
        else:
            instance, label = (self._test_instances[index],
                               self._test_labels[index])
        instance = PIL.Image.fromarray(instance)
        if self._transform is not None:
            instance = self._transform(instance)
        if self._target_transform is not None:
            label = self._target_transform(label)
        return instance, label

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_instances)
        return len(self._test_instances)


class CUB200ReLU43(torch.utils.data.Dataset):
    """CUB200 relu4-3 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/validation data.
        _train_images, list<torch.Tensor>.
        _train_labels, list<int>.
        _val_images, list<torch.Tensor>.
        _val_labels, list<int>.
    """
    def __init__(self, root, train=True, model_path=None):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/validation data.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train

        train_path = os.path.join(self._root,
                                  'ThiNet_Tiny_AutoPruner_relu4-3_train.pth')
        val_path = os.path.join(self._root,
                                'ThiNet_Tiny_AutoPruner_relu4-3_val.pth')
        if not os.path.isfile(train_path) or not os.path.isfile(val_path):
            self._prepareDataset(model_path)

        # Now load the picked data.
        if self._train:
            self._train_images, self._train_labels = torch.load(train_path)
            assert (len(self._train_images) == 5994
                    and len(self._train_labels) == 5994)
        else:
            self._val_images, self._val_labels = torch.load(val_path)
            assert (len(self._val_images) == 5794
                    and len(self._val_labels) == 5794)

    def _prepareDataset(self, model_path):
        """Forward all train/validation images and save their ReLU4-3 feature
        onto disk.

        Args:
            model_path, str.
        """
        import torchvision
        import thinet_autopruner

        print('Prepare CUB-200 ThiNet-Tiny (AutoPruner) ReLU4-3 dataset.')
        assert os.path.isfile(model_path)
        model = thinet_autopruner.ThiNetAutoPrunerTiny(
            model_path=model_path, num_classes=200).cuda()
        model.phase = 'extract'
        model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])
        train_data = CUB200(
            root=self._root, train=True, transform=transform)
        val_data = CUB200(
            root=self._root, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)

        with torch.no_grad():
            train_images, train_labels = self._prepareDatasetHelper(
                is_train=True, model=model, loader=train_loader)
            val_images, val_labels = self._prepareDatasetHelper(
                is_train=False, model=model, loader=val_loader)

        train_path = os.path.join(self._root,
                                  'ThiNet_Tiny_AutoPruner_relu4-3_train.pth')
        val_path = os.path.join(self._root,
                                'ThiNet_Tiny_AutoPruner_relu4-3_val.pth')
        torch.save((train_images, train_labels), train_path)
        torch.save((val_images, val_labels), val_path)

    def _prepareDatasetHelper(self, is_train, model, loader):
        """Extract ReLU4-3 activation.

        This is a help

        Args:
            is_train, bool: Train/Validation.
            model, torch.nn.Module.
            loader, torch.utils.data.DataLoader: Train/Validation loader.

        Returns:
            all_images, list<torch.Tensor<143*28*28>>.
            all_labels, list<int>.
        """
        import tqdm

        all_images = []
        all_labels = []
        for instances, labels in tqdm.tqdm(
                loader, desc='%s' % 'train' if is_train else 'val. '):
            all_labels.append(labels.item())
            instances = instances.cuda()
            instances = model(instances)
            assert instances.size() == (1, 143, 28, 28)
            all_images.append(instances[0].cpu())
        return all_images, all_labels

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            feature, torch.Tensor: relu4-3 feature of the given index.
            target, int: target of the given index.
        """
        if self._train:
            return self._train_images[index], self._train_labels[index]
        return self._val_images[index], self._val_labels[index]

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_images)
        return len(self._val_images)
