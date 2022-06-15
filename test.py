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

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from dataset import chars_convert, collate_fn, ImageDataset
from en_decoder import encoder, decoder
from model import FAN


def load_dataloader() -> DataLoader:
    # Load datasets
    datasets = ImageDataset(dataroot=config.dataroot,
                            annotation_file_name=config.annotation_file_name,
                            image_width=config.model_image_width,
                            image_height=config.model_image_height,
                            mean=config.mean,
                            std=config.std,
                            mode="Test")

    dataloader = DataLoader(dataset=datasets,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=True)

    return dataloader


def build_model(num_classes: int) -> nn.Module:
    # Initialize the model
    model = FAN(num_classes)
    model = model.to(device=config.device)
    print("Build FAN model successfully.")

    # Load the FAN model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load FAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    return model


def main() -> None:
    # Initialize correct predictions image number
    total_correct = 0

    # Generate image labels and model class counts
    chars_list, chars_dict = chars_convert(config.chars_file)
    num_classes = len(chars_dict)

    # Initialize model
    model = build_model(num_classes)

    # Load test dataLoader
    dataloader = load_dataloader()

    # Create a experiment folder results
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Get the number of test image files
    total_files = len(dataloader)

    with open(os.path.join(config.result_dir, config.result_file_name), "w") as f:
        with torch.no_grad():
            for batch_index, (image_path, images, target) in enumerate(dataloader):
                # Create max length prediction
                pred = torch.full([1, config.max_length + 1], 0.0, dtype=torch.long)
                pred_length = torch.IntTensor([config.max_length] * 1)

                # Encode the target
                target, target_length = encoder(target, chars_dict, config.max_length)

                # Transfer in-memory data to CUDA devices to speed up training
                images = images.to(device=config.device, non_blocking=True)
                target = target.to(device=config.device, non_blocking=True)
                target_length = target_length.to(device=config.device, non_blocking=True)
                pred = pred.to(device=config.device, non_blocking=True)
                pred_length = pred_length.to(device=config.device, non_blocking=True)

                # Inference
                output = model(images, pred, pred_length, False)

                # record accuracy
                output = output[:, :target.size(1) - 1, :]

                # Delete [Go] token
                target = target[:, 1:]

                # Decode the target
                _, output_index = output.max(2)
                prediction_chars = decoder(output_index, chars_list, pred_length)
                target_chars = decoder(target[:, 1:], chars_list, target_length)

                # Count correctly predicted characters
                for prediction_char, target_char in zip(prediction_chars, target_chars):
                    # Delete [s] token
                    eos_symbols = prediction_char.find("[s]")
                    prediction_char = prediction_char[:eos_symbols]

                    if prediction_char == target_char:
                        total_correct += 1

                if batch_index < total_files - 1:
                    information = f"`{os.path.basename(image_path[0])}` -> `{prediction_char}`"
                    print(information)
                else:
                    information = f"Acc: {total_correct / total_files * 100:.2f}%"
                    print(information)

                # Text information to be written to the file
                f.write(information + "\n")


if __name__ == "__main__":
    main()
