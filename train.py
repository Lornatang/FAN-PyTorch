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
import shutil
import time
from enum import Enum

import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import chars_convert, collate_fn, ImageDataset
from en_decoder import encoder, decoder
from model import FAN


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc = 0.0

    # Generate image labels and model class counts
    chars_list, chars_dict = chars_convert(config.chars_file)
    num_classes = len(chars_dict)

    train_dataloader, test_dataloader = load_dataset()
    print("Load all datasets successfully.")

    model = build_model(num_classes)
    print("Build FAN model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded pretrained model weights.")

    # Create a folder of experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(model, train_dataloader, chars_dict, criterion, optimizer, epoch, scaler, writer)
        acc = validate(model, test_dataloader, chars_list, chars_dict, epoch, writer, "Test")
        print("\n")

        # Automatically save the model with the highest index
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        torch.save({"epoch": epoch + 1,
                    "best_acc": best_acc,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [DataLoader, DataLoader]:
    # Load train and test datasets
    train_datasets = ImageDataset(dataroot=config.train_dataroot,
                                  annotation_file_name=config.annotation_train_file_name,
                                  image_width=config.model_image_width,
                                  image_height=config.model_image_height,
                                  mean=config.mean,
                                  std=config.std,
                                  mode="Train")
    test_datasets = ImageDataset(dataroot=config.test_dataroot,
                                 annotation_file_name=config.annotation_test_file_name,
                                 image_width=config.model_image_width,
                                 image_height=config.model_image_height,
                                 mean=config.mean,
                                 std=config.std,
                                 mode="Test")

    # Generator all dataloader
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)

    test_dataloader = DataLoader(dataset=test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    return train_dataloader, test_dataloader


def build_model(num_classes: int) -> FAN:
    model = FAN(num_classes)
    model = model.to(device=config.device)

    return model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(model) -> optim.Adadelta:
    optimizer = optim.Adadelta(model.parameters(), config.model_lr, config.model_rho, config.model_eps)

    return optimizer


def train(model: nn.Module,
          train_dataloader: DataLoader,
          chars_dict: dict,
          criterion: nn.CrossEntropyLoss,
          optimizer: optim.Adadelta,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        model (nn.Module): FAN model
        train_dataloader (DataLoader): training dataset iterator
        chars_dict (dict): All char dictionary
        criterion (nn.CrossEntropyLoss): Calculates loss between a continuous time series and a target sequence
        optimizer (optim.Adadelta): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_dataloader)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Get the initialization training time
    end = time.time()

    for batch_index, (_, images, target) in enumerate(train_dataloader):
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Encode the target
        target, _ = encoder(target, chars_dict, config.max_length)

        # Transfer in-memory data to CUDA devices to speed up training
        images = images.to(device=config.device, non_blocking=True)
        target = target.to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images, target[:, :-1], config.max_length, True)
            # cal loss
            loss = criterion(output.view(-1, output.size(-1)), target[:, 1:].contiguous().view(-1))

        # Backpropagation
        scaler.scale(loss).backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.loss_grad_clip_value, config.loss_grad_clip_norm_type)
        # Update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), images.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)


def validate(model: nn.Module,
             dataloader: DataLoader,
             chars_list: list,
             chars_dict: dict,
             epoch: int,
             writer: SummaryWriter,
             mode: str) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): CRNN model
        dataloader (DataLoader): Test dataset iterator
        chars_list (dict): All char list
        chars_dict (dict): All char dictionary
        epoch (int): Number of test epochs during training of the adversarial network
        writer (SummaryWriter): Log file management function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize correct predictions image number
    total_correct = 0
    total_files = 0

    with torch.no_grad():
        for batch_index, (_, images, target) in enumerate(dataloader):
            # Get image batch size
            batch_size = images.size(0)

            # Get how many data the current batch has and increase the total number of tests
            total_files += batch_size

            # Create max length prediction
            pred = torch.full([batch_size, config.max_length + 1], 0.0, dtype=torch.long)
            pred_length = torch.IntTensor([config.max_length] * batch_size)

            # Encode the target
            target, target_length = encoder(target, chars_dict, config.max_length)

            # Transfer in-memory data to CUDA devices to speed up training
            images = images.to(device=config.device, non_blocking=True)
            target = target.to(device=config.device, non_blocking=True)
            target_length = target_length.to(device=config.device, non_blocking=True)
            pred = pred.to(device=config.device, non_blocking=True)
            pred_length = pred_length.to(device=config.device, non_blocking=True)

            # Mixed precision testing
            with amp.autocast():
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
                if prediction_char == target_char:
                    total_correct += 1

    # print metrics
    acc = (total_correct / total_files) * 100
    print(f"* Acc: {acc:.2f}%")

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc", acc, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
