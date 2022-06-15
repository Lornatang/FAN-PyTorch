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
import argparse

import cv2
import numpy as np
import torch

import config
import imgproc
from dataset import chars_convert
from en_decoder import decoder
from model import FAN


def main(args):
    # Generate image labels and model class counts
    chars_list, chars_dict = chars_convert(config.chars_dict_path)
    num_classes = len(chars_dict)

    # Initialize the model
    model = FAN(num_classes)
    model = model.to(device=config.device)
    print("Build FAN model successfully.")

    # Load the FAN model weights
    checkpoint = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load FAN model weights `{args.weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read the image and convert it to grayscale
    image = cv2.imread(args.image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Scale to the size of the image that the model can accept
    gray_image = cv2.resize(gray_image,
                            (config.model_image_width, config.model_image_height),
                            interpolation=cv2.INTER_CUBIC)
    gray_image = np.reshape(gray_image, (config.model_image_height, config.model_image_width, 1))

    # Normalize and convert to Tensor format
    gray_tensor = imgproc.image2tensor(gray_image, mean=config.mean, std=config.std).unsqueeze_(0)

    # Create max length prediction
    pred = torch.full([1, config.max_length + 1], 0.0, dtype=torch.long)
    pred_length = torch.IntTensor([config.max_length] * 1)

    # Transfer in-memory data to CUDA devices to speed up training
    gray_tensor = gray_tensor.to(device=config.device, non_blocking=True)
    pred = pred.to(device=config.device, non_blocking=True)
    pred_length = pred_length.to(device=config.device, non_blocking=True)

    # Inference
    with torch.no_grad():
        output = model(gray_tensor, pred, pred_length, False)

        # Decode the target
        _, output_index = output.max(2)
        prediction_chars = decoder(output_index, chars_list, pred_length)

        for prediction_char in prediction_chars:
            # Delete [s] token
            eos_symbols = prediction_char.find("[s]")
            prediction_char = prediction_char[:eos_symbols]

            print(f"`{args.image_path}` -> `{prediction_char}`\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAN model predicts character content in images.")
    parser.add_argument("--image_path", type=str, help="Character image address to be tested.")
    parser.add_argument("--weights_path", type=str, help="Model weight file path.")
    args = parser.parse_args()

    main(args)
