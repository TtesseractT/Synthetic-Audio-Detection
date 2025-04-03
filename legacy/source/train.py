#!/usr/bin/env python3
# Company:  Uhmbrella Ltd 2025
# Author:   Sabian Hibbs
# Date:     2025-01-01
# Version:  1.0
# License:  MIT


import os
import sys
import argparse
import logging
import warnings
import random
import numpy as np
from datetime import datetime
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import torch.nn.init as init

# Suppress warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--data-dir', default='./dataset', type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=30, type=int, help='Number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')  # Adjusted
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=42, type=int, help='Seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--resume', default='./checkpoints', type=str, help='Path to resume checkpoint')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on validation set')

    # Define available models
    available_models = timm.list_models('resnet*')
    parser.add_argument('--model-name', default='resnet151', type=str,
                        choices=available_models, help='Name of model to use')
    return parser.parse_args()


def setup_logging():
    """Sets up logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/training_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


class SpectrogramDataset(Dataset):
    """Custom dataset for loading audio files and converting them to spectrograms."""

    def __init__(self, data_dir, mode, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            mode (str): Mode of the dataset ('train' or 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        self.classes = ['Real', 'class1', 'class2', 'class3', 'class4']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset(data_dir)

        logging.info(f"Found {len(self.samples)} samples for mode {self.mode}")
        logging.info(f"Classes: {self.classes}")
        logging.info(f"Class-to-Index Mapping: {self.class_to_idx}")

        # Define mel spectrogram and amplitude transformations
        self.mel_spec = MelSpectrogram(
            sample_rate=32000,  # Adjusted sample rate to 32kHz
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=12000, # Adjusted f_max to 16kHz (can be lower than sample_rate/2) i.e 12000
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=80)

        # Initialize data augmentation transforms
        if self.mode == 'train':
            self.augmentation_transforms = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35)
            )
        else:
            self.augmentation_transforms = None

        self.min_length_ratio = 0.9

    def _make_dataset(self, directory):
        """Generates a list of file paths and their respective class labels."""
        instances = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(directory, self.mode, target_class)
            if not os.path.isdir(target_dir):
                logging.warning(f"Directory {target_dir} does not exist. Skipping.")
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith('.wav'):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        instances.append(item)

        if not instances:
            raise RuntimeError(f"No wav files found in {directory}/{self.mode}")

        return instances

    def __getitem__(self, index):
        """Loads an audio file, processes it, and returns the spectrogram and label."""
        path, target = self.samples[index]
        try:
            waveform, sample_rate = torchaudio.load(path)

            # Skip if waveform is empty
            if waveform.numel() == 0:
                logging.debug(f"Empty waveform detected at index {index} for path {path}")
                return None

            # Resample to 32kHz if necessary
            if sample_rate != 32000:
                resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
                waveform = resample(waveform)

            # Define segment length (4 seconds at 32kHz)
            segment_length = 4 * 32000
            required_length = segment_length * 2

            # Check if waveform length is acceptable
            if waveform.size(1) >= required_length:
                # Standard processing for full-length files
                wave_segments = [
                    waveform[:, :segment_length],
                    waveform[:, segment_length:2 * segment_length]
                ]

            elif waveform.size(1) >= segment_length:
                # If we have at least one segment length, duplicate it
                logging.debug(f"Using single segment duplication for {path}")
                first_segment = waveform[:, :segment_length]
                wave_segments = [first_segment, first_segment]

            elif waveform.size(1) >= segment_length * self.min_length_ratio:
                # If we have close to one segment length, pad it and duplicate
                logging.debug(f"Padding shorter file {path}")
                padding_needed = segment_length - waveform.size(1)
                padded_segment = torch.nn.functional.pad(
                    waveform, (0, padding_needed), mode='constant', value=0
                )
                wave_segments = [padded_segment, padded_segment]

            else:
                logging.debug(
                    f"File too short at index {index}, path {path}. "
                    f"Length: {waveform.size(1)}, Required: {required_length}"
                )
                return None

            processed_segments = []
            for wave_seg in wave_segments:
                spec = self.mel_spec(wave_seg)
                spec = self.amplitude_to_db(spec)

                # Apply data augmentation
                if self.mode == 'train' and self.augmentation_transforms:
                    spec = self.augmentation_transforms(spec)

                # Normalize per spectrogram
                spec = (spec - spec.mean()) / (spec.std() + 1e-6)
                spec = transforms.Resize((512, 512))(spec)  # Changed from (224, 224) to (512, 512)

                # Convert to 3 channels
                spec = spec.repeat(3, 1, 1)

                if self.transform:
                    spec = self.transform(spec)

                processed_segments.append(spec)

            return processed_segments[0], target, processed_segments[1], target

        except Exception as e:
            logging.warning(f"Error processing file at index {index}, path {path}: {str(e)}")
            return None

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)


def custom_collate_fn(batch):
    """Custom collate function to filter out None values from the batch."""
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None

    # Separate the batch elements
    input1, target1, input2, target2 = zip(*batch)

    # Convert to tensors
    input1 = torch.stack(input1)
    input2 = torch.stack(input2)
    target1 = torch.tensor(target1)
    target2 = torch.tensor(target2)

    return input1, target1, input2, target2


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, writer, total_steps, device):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}]")

    for batch_idx, batch in loop:
        if batch is None:
            continue  # Skip this batch if __getitem__ returned None
        input1, target1, input2, target2 = batch
        # Concatenate the two segments to create a batch
        inputs = torch.cat((input1, input2), dim=0).to(device, non_blocking=True)
        targets = torch.cat((target1, target2), dim=0).to(device, non_blocking=True)

        optimizer.zero_grad()

        try:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Check for NaN or Inf in the loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(
                    f'NaN or Inf loss encountered at epoch {epoch}, '
                    f'batch {batch_idx}, skipping step.'
                )
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            running_loss += loss.item() * inputs.size(0)

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.*correct/total:.2f}%",
                lr=f"{optimizer.param_groups[-1]['lr']:.6f}"
            )

            total_steps += 1

            if total_steps % 100 == 0:
                writer.add_scalar('Loss/train_step', loss.item(), total_steps)
                writer.add_scalar('Accuracy/train_step', 100.*correct/total, total_steps)
                writer.add_scalar('Learning_rate', optimizer.param_groups[-1]['lr'], total_steps)

        except Exception as e:
            logging.error(f'Error in training batch {batch_idx}: {str(e)}')
            continue

    epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
    epoch_acc = 100. * correct / total if total > 0 else 0.0

    # Adjust scheduler if using ReduceLROnPlateau
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(epoch_loss)
    else:
        scheduler.step()

    return epoch_loss, epoch_acc, total_steps


def validate(args, val_loader, model, criterion, epoch, device):
    """Performs one epoch of validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []
    classes = val_loader.dataset.classes  # Dynamically determine class names

    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation [{epoch}]")

    with torch.no_grad():
        for batch_idx, batch in loop:
            if batch is None:
                continue

            # Unpack all four values
            input1, target1, input2, target2 = batch

            # Concatenate the inputs and targets like in training
            inputs = torch.cat((input1, input2), dim=0).to(device, non_blocking=True)
            targets = torch.cat((target1, target2), dim=0).to(device, non_blocking=True)

            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)

                # Ensure predicted labels are within expected range
                predicted = predicted.clamp(min=0, max=len(classes)-1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100.*correct/total:.2f}%"
                )

            except Exception as e:
                logging.error(f'Error in validation batch {batch_idx}: {str(e)}')
                continue

    epoch_loss = running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    epoch_acc = 100. * correct / total if total > 0 else 0.0

    # Debugging: Log unique classes
    unique_targets = set(all_targets)
    unique_predictions = set(all_predictions)
    logging.info(f"Unique targets in validation: {unique_targets}")
    logging.info(f"Unique predictions in validation: {unique_predictions}")

    # Generate classification report dynamically
    from sklearn.metrics import classification_report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=classes,
        labels=list(range(len(classes)))
    )
    logging.info(f"\nClassification Report:\n{report}")

    return epoch_loss, epoch_acc, all_predictions, all_targets


def evaluate(args, model, val_loader, criterion, device):
    """Evaluates the model on the validation set and prints detailed metrics."""
    model.eval()

    all_predictions = []
    all_targets = []

    # 2) Updated to 5 classes for evaluation
    classes = ['Real', 'class1', 'class2', 'class3', 'class4']
    class_correct = [0] * 5
    class_total = [0] * 5

    logging.info("Starting evaluation...")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch is None:
                continue

            # Unpack all four values
            input1, target1, input2, target2 = batch

            # Concatenate the inputs and targets
            inputs = torch.cat((input1, input2), dim=0).to(device, non_blocking=True)
            targets = torch.cat((target1, target2), dim=0).to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Per-class accuracy
            correct = predicted.eq(targets)
            for i in range(len(targets)):
                label = targets[i]
                if label < len(classes):  # Ensure label is within range
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

    # Calculate overall accuracy
    accuracy = 100 * sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0.0

    # Print results
    logging.info("\nEvaluation Results:")
    logging.info(f"Overall Accuracy: {accuracy:.2f}%\n")
    logging.info("Per-class Accuracy:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            logging.info(f"{classes[i]}: {class_acc:.2f}% ({int(class_correct[i])}/{class_total[i]})")
        else:
            logging.info(f"{classes[i]}: No samples.")

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(
        all_targets, all_predictions,
        labels=list(range(len(classes)))
    )
    report = classification_report(
        all_targets, all_predictions,
        target_names=classes,
        labels=list(range(len(classes)))
    )

    logging.info("\nConfusion Matrix:")
    logging.info(f"{cm}")

    logging.info("\nDetailed Classification Report:")
    logging.info(report)


def get_dataloaders(args):
    """Creates data loaders for training and validation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # Changed to 512
        # We already normalize in the dataset
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Changed to 512
        # We already normalize in the dataset
    ])

    train_dataset = SpectrogramDataset(
        data_dir=args.data_dir,
        mode='train',
        transform=train_transform,
    )

    val_dataset = SpectrogramDataset(
        data_dir=args.data_dir,
        mode='test',
        transform=val_transform,
    )

    # Adjust batch size for multi-GPU
    if args.num_gpus > 0:
        total_batch_size = args.batch_size * args.num_gpus
    else:
        total_batch_size = args.batch_size

    logging.info(f"Total batch size: {total_batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=total_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=total_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader


def initialize_weights(model):
    """Initializes the weights of the model."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)


def get_model(model):
    """Returns the underlying model, handling DataParallel."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    else:
        return model


def main():
    """Main function to run the training and evaluation."""
    args = parse_args()
    setup_logging()
    logging.info(f"Arguments: {args}")

    if args.num_gpus > torch.cuda.device_count():
        logging.error(
            f"Requested number of GPUs ({args.num_gpus}) "
            f"is greater than available GPUs ({torch.cuda.device_count()})"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Using {args.num_gpus} GPUs")

    if args.num_gpus > 0:
        total_batch_size = args.batch_size * args.num_gpus
    else:
        total_batch_size = args.batch_size
    logging.info(f"Total batch size: {total_batch_size}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.info("Creating model with RANDOM weights...") # Log message updated
    # Set pretrained=False to initialize with random weights instead of ImageNet weights
    model = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # 3) Updated classification head for 5 classes
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(model.num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5)  # Now 5 classes instead of 4
    )

    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze the last stage
    model_to_update = get_model(model)
    for param in model_to_update.layer4.parameters():
        param.requires_grad = True

    model = model.to(device)

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
        logging.info("Model wrapped with DataParallel")

    # Data loaders
    train_loader, val_loader = get_dataloaders(args)
    logging.info(f"Number of training samples: {len(train_loader.dataset)}")
    logging.info(f"Number of validation samples: {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    writer = SummaryWriter(log_dir=f'runs/experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    best_acc = 0.0
    total_steps = 0
    start_epoch = 0

    # Checkpoint loading
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            state_dict = checkpoint['state_dict']
            get_model(model).load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            total_steps = checkpoint.get('total_steps', 0)
            logging.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.error(f"No checkpoint found at '{args.resume}'")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f'\nEpoch: {epoch}/{args.epochs - 1}')

        if epoch == args.epochs // 3:
            logging.info("Unfreezing more layers...")
            # Unfreeze the second-to-last layer
            for param in model_to_update.layer3.parameters():
                param.requires_grad = True

        train_loss, train_acc, total_steps = train(
            args, train_loader, model, criterion, optimizer, scheduler,
            epoch, writer, total_steps, device
        )

        val_loss, val_acc, _, _ = validate(args, val_loader, model, criterion, epoch, device)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # Save checkpoint for every epoch
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}_acc_{val_acc:.2f}.pth')
        state_dict = get_model(model).state_dict()
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'total_steps': total_steps
        }, checkpoint_path)
        
        # Log differently if it's the best model so far
        if is_best:
            logging.info(f'Saved best model with accuracy: {val_acc:.2f}%')
        else:
            logging.info(f'Saved checkpoint for epoch {epoch} with accuracy: {val_acc:.2f}%')

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)

    writer.close()
    logging.info('Training completed.')
    logging.info(f'Best validation accuracy: {best_acc:.2f}%')

    if args.evaluate:
        evaluate(args, model, val_loader, criterion, device)


if __name__ == '__main__':
    main()
