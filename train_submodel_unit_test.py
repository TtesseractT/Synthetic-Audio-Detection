import os
import sys
import shutil
import tempfile
import unittest
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np

# Import the training script as a module
import train_submodel

# Dummy model to be used in place of timm's model during tests.
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Set a dummy num_features attribute required by the head creation.
        self.num_features = 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512 * 3, 2)
        )

    def forward(self, x):
        # For testing, simply forward through the head.
        return self.head(x)

# Dummy replacement for timm.create_model to avoid downloading pretrained weights.
def dummy_create_model(model_name, pretrained, num_classes):
    return DummyModel()

class TestTrainSubmodel(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to mimic a dataset structure.
        self.test_dir = tempfile.mkdtemp()
        for mode in ['train', 'test']:
            for cls in ['Real', 'Class1']:
                os.makedirs(os.path.join(self.test_dir, mode, cls), exist_ok=True)

        # Create a dummy .wav file (8 seconds long at 32000 Hz) in each class directory.
        sample_rate = 32000
        duration_seconds = 8  # 8 seconds (provides two full segments)
        waveform = torch.rand(1, sample_rate * duration_seconds)
        for mode in ['train', 'test']:
            for cls in ['Real', 'Class1']:
                file_path = os.path.join(self.test_dir, mode, cls, "dummy.wav")
                torchaudio.save(file_path, waveform, sample_rate)

        # Monkey-patch timm.create_model to return our dummy model.
        self.original_create_model = train_submodel.timm.create_model
        train_submodel.timm.create_model = dummy_create_model

        # Set up a minimal args Namespace for testing.
        self.args = train_submodel.argparse.Namespace(
            data_dir=self.test_dir,
            batch_size=1,
            epochs=1,
            lr=0.001,
            workers=0,
            seed=42,
            gpu=0,
            num_gpus=0,  # Force CPU for testing.
            checkpoint_dir=os.path.join(self.test_dir, "checkpoints"),
            resume="",
            evaluate=False,
            Class0="Real",
            Class1="Class1",
            model_name="resnet18"
        )

    def tearDown(self):
        # Remove the temporary directory.
        shutil.rmtree(self.test_dir)
        # Restore the original timm.create_model.
        train_submodel.timm.create_model = self.original_create_model
        # Remove logs if created.
        if os.path.exists("logs"):
            shutil.rmtree("logs", ignore_errors=True)

    def test_parse_args(self):
        # Test that parse_args returns default values when no extra arguments are provided.
        test_args = ["prog"]
        original_argv = sys.argv
        sys.argv = test_args
        try:
            args = train_submodel.parse_args()
            self.assertEqual(args.data_dir, "./dataset")
        finally:
            sys.argv = original_argv

    def test_setup_logging(self):
        # Test that the logging setup creates a logs directory.
        train_submodel.setup_logging()
        self.assertTrue(os.path.exists("logs"))

    def test_spectrogram_dataset(self):
        # Test the SpectrogramDataset __getitem__ functionality.
        dataset = train_submodel.SpectrogramDataset(
            data_dir=self.test_dir,
            mode='train',
            transform=None,
            class_names=["Real", "Class1"]
        )
        # Get the first sample.
        item = dataset[0]
        self.assertIsNotNone(item)
        # Expecting a tuple of 4 elements: (spec1, target1, spec2, target2)
        self.assertEqual(len(item), 4)
        spec1, target1, spec2, target2 = item
        self.assertTrue(torch.is_tensor(spec1))
        self.assertTrue(torch.is_tensor(spec2))
        self.assertIsInstance(target1, int)
        self.assertIsInstance(target2, int)

    def test_custom_collate_fn(self):
        # Create a dummy batch with one valid sample and one None entry.
        dummy_tensor = torch.rand(3, 512, 512)
        sample = (dummy_tensor, 0, dummy_tensor, 0)
        batch = [sample, None]
        collated = train_submodel.custom_collate_fn(batch)
        # Check that None entries are filtered out.
        self.assertIsNotNone(collated)
        input1, target1, input2, target2 = collated
        self.assertEqual(input1.shape[0], 1)

    def test_train_function(self):
        # Create a dummy DataLoader (as a list) containing one sample.
        dummy_tensor = torch.rand(3, 512, 512)
        sample = (dummy_tensor, 0, dummy_tensor, 0)
        data_loader = [sample]
        # Set up a dummy model and other training components.
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        writer = train_submodel.SummaryWriter(log_dir=self.test_dir)
        device = torch.device("cpu")
        total_steps = 0

        train_loss, train_acc, total_steps = train_submodel.train(
            self.args, data_loader, model, criterion, optimizer, scheduler, 0, writer, total_steps, device
        )
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        writer.close()

    def test_validate_function(self):
        # Create a dummy DataLoader (list) with one sample.
        dummy_tensor = torch.rand(3, 512, 512)
        sample = (dummy_tensor, 0, dummy_tensor, 0)
        data_loader = [sample]
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        val_loss, val_acc, predictions, targets = train_submodel.validate(
            self.args, data_loader, model, criterion, 0, device
        )
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        # Because the __getitem__ in validation concatenates two segments.
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(targets), 2)

    def test_evaluate_function(self):
        # Create a dummy DataLoader (list) with one sample.
        dummy_tensor = torch.rand(3, 512, 512)
        sample = (dummy_tensor, 0, dummy_tensor, 0)
        data_loader = [sample]
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        # Run evaluate and ensure no exception is raised.
        try:
            train_submodel.evaluate(self.args, model, data_loader, criterion, device)
        except Exception as e:
            self.fail(f"evaluate() raised an Exception: {e}")

    def test_initialize_weights(self):
        # Build a dummy model containing Conv2d and Linear layers.
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.Linear(16, 2)
        )
        orig_conv_weight = model[0].weight.clone()
        orig_linear_weight = model[1].weight.clone()
        train_submodel.initialize_weights(model)
        # Verify that the weights have been re-initialized.
        self.assertFalse(torch.equal(model[0].weight, orig_conv_weight))
        self.assertFalse(torch.equal(model[1].weight, orig_linear_weight))

    def test_get_model(self):
        # Wrap a dummy model in DataParallel and test get_model.
        model = DummyModel()
        dp_model = torch.nn.DataParallel(model)
        unwrapped = train_submodel.get_model(dp_model)
        self.assertIs(unwrapped, model)

    def test_get_dataloaders(self):
        # Test that get_dataloaders returns non-empty DataLoader objects.
        self.args.num_gpus = 0  # Ensure CPU is used.
        train_loader, val_loader = train_submodel.get_dataloaders(self.args)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)

    def test_main_integration(self):
        # Integration test for main(). Override parse_args to return our test args.
        args = self.args
        args.epochs = 1  # Run only one epoch.
        original_parse_args = train_submodel.parse_args
        train_submodel.parse_args = lambda: args
        try:
            # Call main and ensure it completes without error.
            train_submodel.main()
        except Exception as e:
            self.fail(f"main() raised an Exception unexpectedly: {e}")
        finally:
            train_submodel.parse_args = original_parse_args

if __name__ == '__main__':
    unittest.main()
