import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

import train_submodel

# ------------------------------------------------------------------
# 1) A "DummyResnet" that includes layer3, layer4, and a head inside forward()
# ------------------------------------------------------------------
class DummyResnet(nn.Module):
    def __init__(self, num_classes=2):
        super(DummyResnet, self).__init__()
        self.num_features = 64
        # Provide a dummy layer3 and layer4 so that partial unfreezes in main() won’t fail
        self.layer3 = nn.Sequential(nn.Identity())
        self.layer4 = nn.Sequential(nn.Identity())

        # Final head: produce logits of shape [batch, num_classes]
        # from the 64x7x7 feature map
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # shape => [batch, 64, 1, 1]
            nn.Flatten(),                 # shape => [batch, 64]
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)   # shape => [batch, num_classes]
        )

    def forward(self, x):
        # x is shape [batch, 3, 512, 512] after cat-ing the segments
        # Let’s pretend we do something with layer3, layer4
        x = self.layer3(x)
        x = self.layer4(x)

        # For demonstration, create a fake feature map of shape [batch, 64, 7, 7].
        # (In real usage, you’d have actual conv layers.)
        batch_size = x.size(0)
        feats = torch.zeros(batch_size, 64, 7, 7, device=x.device)

        # Now pass feats through the classification head => shape [batch, num_classes]
        out = self.head(feats)
        return out


# ------------------------------------------------------------------
# 2) Monkey-patch timm.create_model to return our DummyResnet
# ------------------------------------------------------------------
def dummy_create_model(model_name, pretrained, num_classes):
    return DummyResnet(num_classes=2)

# ------------------------------------------------------------------
# 3) Monkey-patch torchaudio.load so no real files are needed
# ------------------------------------------------------------------
def dummy_torchaudio_load(path):
    sample_rate = 32000
    # Return an 8s random waveform => 2 segments each 4s
    waveform = torch.rand(1, 256000)
    return waveform, sample_rate

# ------------------------------------------------------------------
# 4) Monkey-patch SpectrogramDataset._make_dataset to skip filesystem
# ------------------------------------------------------------------
def dummy_make_dataset(self, directory):
    # Return a single dummy entry with class label 0
    return [("dummy_path.wav", 0)]

# ------------------------------------------------------------------
# 5) A tiny Dataset that yields (input1, target1, input2, target2)
# ------------------------------------------------------------------
class DummyDataset(Dataset):
    def __init__(self, num_items=1, classes=None):
        self.num_items = num_items
        if classes is None:
            classes = ["Real", "Class1"]
        self.classes = classes

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        """
        Return (input1, target1, input2, target2)
        - input1, input2 => shape [3, 512, 512]
        - target1, target2 => shape [1]
        """
        input1 = torch.rand(3, 512, 512)
        input2 = torch.rand(3, 512, 512)
        # Wrap targets as 1D => shape [1]
        target1 = torch.tensor([0])
        target2 = torch.tensor([0])
        return (input1, target1, input2, target2)


# ------------------------------------------------------------------
# 6) Our Unit Tests
# ------------------------------------------------------------------
class TestTrainSubmodelNoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Monkey-patch timm.create_model, torchaudio.load, and the dataset-building method
        cls.original_create_model = train_submodel.timm.create_model
        train_submodel.timm.create_model = dummy_create_model

        cls.original_torchaudio_load = torchaudio.load
        torchaudio.load = dummy_torchaudio_load

        cls.original_make_dataset = train_submodel.SpectrogramDataset._make_dataset
        train_submodel.SpectrogramDataset._make_dataset = dummy_make_dataset

        # Minimal arguments for testing
        cls.args = train_submodel.argparse.Namespace(
            data_dir="dummy_dir",
            batch_size=1,
            epochs=1,
            lr=0.001,
            workers=0,
            seed=42,
            gpu=0,
            num_gpus=0,  # Force CPU
            checkpoint_dir="dummy_checkpoints",
            resume="",
            evaluate=False,
            Class0="Real",
            Class1="Class1",
            model_name="resnet18"
        )

    @classmethod
    def tearDownClass(cls):
        # Restore patched functions
        train_submodel.timm.create_model = cls.original_create_model
        torchaudio.load = cls.original_torchaudio_load
        train_submodel.SpectrogramDataset._make_dataset = cls.original_make_dataset

        # Remove logs if any
        if os.path.exists("logs"):
            import shutil
            shutil.rmtree("logs", ignore_errors=True)

    def test_parse_args(self):
        # Make sure parse_args works with no extra CLI arguments
        test_args = ["prog"]
        original_argv = sys.argv
        sys.argv = test_args
        try:
            args = train_submodel.parse_args()
            self.assertEqual(args.data_dir, "./dataset")
        finally:
            sys.argv = original_argv

    def test_setup_logging(self):
        train_submodel.setup_logging()
        self.assertTrue(os.path.exists("logs"))

    def test_spectrogram_dataset(self):
        dataset = train_submodel.SpectrogramDataset(
            data_dir=self.args.data_dir,
            mode="train",
            transform=None,
            class_names=["Real", "Class1"]
        )
        sample = dataset[0]
        self.assertIsNotNone(sample)
        # Should be (spec1, tgt1, spec2, tgt2)
        self.assertEqual(len(sample), 4)

    def test_custom_collate_fn(self):
        # Collate function expects list of tuples => (input1, target1, input2, target2)
        # and merges them
        sample = (
            torch.rand(3,512,512),
            torch.tensor([0]),
            torch.rand(3,512,512),
            torch.tensor([0])
        )
        batch = [sample, None]  # includes a None
        collated = train_submodel.custom_collate_fn(batch)
        self.assertIsNotNone(collated)
        input1, target1, input2, target2 = collated
        self.assertEqual(input1.shape, (1,3,512,512))
        self.assertEqual(target1.shape, (1,))

    def test_train_function(self):
        # Use a PyTorch DataLoader with our DummyDataset
        dummy_dataset = DummyDataset(num_items=1, classes=["Real", "Class1"])
        dummy_loader = DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=train_submodel.custom_collate_fn
        )

        model = DummyResnet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        writer = train_submodel.SummaryWriter(log_dir="dummy_runs")
        device = torch.device("cpu")
        total_steps = 0

        train_loss, train_acc, total_steps = train_submodel.train(
            self.args, dummy_loader, model, criterion, optimizer,
            scheduler, 0, writer, total_steps, device
        )
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)

        writer.close()

    def test_validate_function(self):
        dummy_dataset = DummyDataset(num_items=1, classes=["Real", "Class1"])
        dummy_loader = DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=train_submodel.custom_collate_fn
        )
        model = DummyResnet()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        val_loss, val_acc, predictions, targets = train_submodel.validate(
            self.args, dummy_loader, model, criterion, 0, device
        )
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)

    def test_evaluate_function(self):
        dummy_dataset = DummyDataset(num_items=1, classes=["Real", "Class1"])
        dummy_loader = DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=train_submodel.custom_collate_fn
        )
        model = DummyResnet()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        try:
            train_submodel.evaluate(self.args, model, dummy_loader, criterion, device)
        except Exception as e:
            self.fail(f"evaluate() raised an Exception: {e}")

    def test_initialize_weights(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.Linear(16, 2)
        )
        orig_conv = model[0].weight.clone()
        orig_linear = model[1].weight.clone()
        train_submodel.initialize_weights(model)
        self.assertFalse(torch.equal(model[0].weight, orig_conv))
        self.assertFalse(torch.equal(model[1].weight, orig_linear))

    def test_get_model(self):
        model = DummyResnet()
        dp_model = nn.DataParallel(model)
        unwrapped = train_submodel.get_model(dp_model)
        self.assertIs(unwrapped, model)

    def test_get_dataloaders(self):
        # Just ensure it returns two loaders. They won't match our dummy dataset, but that's okay.
        train_loader, val_loader = train_submodel.get_dataloaders(self.args)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

    def test_main_integration(self):
        # Patch parse_args so main() uses our dummy config
        self.args.epochs = 1
        original_parse_args = train_submodel.parse_args
        train_submodel.parse_args = lambda: self.args
        try:
            train_submodel.main()
        except Exception as e:
            self.fail(f"main() raised an Exception unexpectedly: {e}")
        finally:
            train_submodel.parse_args = original_parse_args


if __name__ == "__main__":
    unittest.main()
