# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset
import pathlib
from torchvision.datasets import folder as dataset_parser

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class MyUnlabeledDataset(Dataset):
    def __init__(self, dataset_root, split, transform,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        file_list = split[0]
        path_list = split[1]

        lines = []
        for file, path in zip(file_list, path_list):
            with open(os.path.join(self.dataset_root, file), 'r') as f:
                line = f.readlines()
                # prepend the path to the each line !!!
                line = [os.path.join(path, l) for l in line]
            lines.extend(line)

        self.data = []
        self.labels = []
        for line in lines:
            path, id, is_fewshot = line.strip('\n').split(' ')
            file_path = path
            self.data.append((file_path, int(id), int(is_fewshot)))
            self.labels.append(int(id))
        
        self.targets = self.labels  # Sampler needs to use targets

        self.transform = transform
        print(f'# of images in {split}: {len(self.data)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        img = self.transform(img) 

        return img, label
