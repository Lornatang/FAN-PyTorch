# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import imgproc

__all__ = [
    "chars_convert", "collate_fn", "ImageDataset"
]


def chars_convert(chars_file: str) -> dict:
    # Create empty label string
    chars = ""

    # Combine tag characters all into a single tag string
    with open(chars_file, "r") as f:
        for char in f.readlines():
            char = char.strip("\n")
            chars += char

    # Add reserved characters. [Go]: start token. [s]:end token.
    chars = list(chars)
    chars = ["[Go]", "[s]"] + chars

    # Convert a single tag string to list
    chars_list = list(chars)
    # Convert a single tag string to dictionary
    chars_dict = {char: i for i, char in enumerate(chars)}

    return chars_list, chars_dict


def collate_fn(batch: [str, torch.Tensor, list]) -> [str, torch.Tensor, list]:
    image_path, images, target = zip(*batch)
    images = torch.stack(images, 0)

    return image_path, images, target


class ImageDataset(Dataset):
    def __init__(self,
                 dataroot: str = None,
                 annotation_file_name: str = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        self.dataroot = dataroot
        self.annotation_file_name = annotation_file_name
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std
        self.mode = mode

        self.images_path, self.images_target = self.load_image_label_from_file()

    def load_image_label_from_file(self):
        # Initialize the definition of image path, image text information, etc.
        images_path = []
        images_target = []

        # Read image path and corresponding text information
        with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
            for line in f.readlines():
                image_path, image_target = line.strip().split(" ")
                images_path.append(os.path.join(self.dataroot, image_path))
                images_target.append(image_target)

        return images_path, images_target

    def __getitem__(self, index: int) -> [str, torch.Tensor, str]:
        image_path = self.images_path[index]
        target = self.images_target[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        image = imgproc.image2tensor(image, mean=self.mean, std=self.std)

        if self.mode == "Train" or self.mode == "Valid" or self.mode == "Test":
            return image_path, image, target
        else:
            raise ValueError("Unsupported data processing model, please use `Train`, `Valid` or `Test`.")

    def __len__(self) -> int:
        return len(self.images_path)
