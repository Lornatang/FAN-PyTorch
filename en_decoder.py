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
import torch

__all__ = [
    "encoder", "decoder"
]


def encoder(chars: str, chars_dict: dict, max_length: int) -> [torch.LongTensor, torch.IntTensor]:
    """Encode chars function

    Args:
        chars (str): Str of chars
        chars_dict (dict): Dict of char
        max_length (int): Max length of text target

    Returns:
        encoder_chars (torch.LongTensor): Encoded chars
        chars_length (torch.IntTensor): Encoded chars length

    """
    chars_length = [len(s) + 1 for s in chars]
    # Add [Go] token
    max_length += 1
    encoder_chars = torch.LongTensor(len(chars_length), max_length + 1).zero_()

    for i, char in enumerate(chars):
        chars = list(char)
        chars.append("[s]")
        chars = [chars_dict[char] for char in chars]
        encoder_chars[i][1:1 + len(chars)] = torch.LongTensor(chars)

    chars_length = torch.IntTensor(chars_length)

    return encoder_chars, chars_length


def decoder(output_index: torch.Tensor, chars_list: list, chars_length: torch.Tensor) -> list:
    """

    Args:
        output_index (torch.Tensor): Select max probability then decode index to character
        chars_list (dict): List of char
        chars_length (torch.Tensor): The target chars length

    Returns:
        decoder_chars (list): Decoder chars

    """
    decoder_chars = []
    for char_index, _ in enumerate(chars_length):
        decoder_char = "".join([chars_list[i] for i in output_index[char_index, :]])
        decoder_chars.append(decoder_char)

    return decoder_chars
