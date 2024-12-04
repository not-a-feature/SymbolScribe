import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import csv
from symbols import symbols
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from model import SymbolCNN, crop_to_content

label_mapping = {symbol[0]: i for i, symbol in enumerate(symbols)}

# Dataset configuration
image_size = (32, 32)


class SymbolDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        crop_to_content=False,
        load_directly=False,
        num_workers=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_to_content = crop_to_content
        self.load_directly = load_directly
        self.num_workers = num_workers if num_workers else os.cpu_count() // 2

        self.labels = [label_mapping[label] for label in self.data.iloc[:, 1]]
        print(f"Dataset with {len(self.data)} samples.")
        if load_directly:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.loaded_data = list(
                    tqdm(
                        executor.map(self.load, range(len(self.data))),
                        total=len(self.data),
                        desc="Loading Images",
                    )
                )
        else:
            print("Data will be loaded on demand.")

    def load(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        try:
            image = Image.open(img_name)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, 0, 0, None  # Return None for both image and label on failure

        if self.crop_to_content:
            image = crop_to_content(image)

        width, height = image.size

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        # print("loaded", label, idx)
        return image, width, height, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_directly:
            return self.loaded_data[idx]
        else:
            return self.load(idx)


def log_metrics(metrics, epoch, metrics_path):
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        writer.writeheader()
        for i in range(epoch + 1):
            row = {k: v[i] for k, v in metrics.items()}
            writer.writerow(row)


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.002):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_top5": [],
        "test_top5": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        running_loss, correct_train, total_train, top5_train = 0.0, 0, 0, 0
        for images, widths, heights, labels in train_loader:
            images = images.to(device)
            widths, heights = widths.to(device), heights.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, widths, heights)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.topk(5, dim=1)
            correct_train += (predicted[:, 0] == labels).sum().item()
            total_train += labels.size(0)
            top5_train += (predicted == labels.view(-1, 1)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_top5s = top5_train / total_train

        # Testing
        model.eval()
        test_loss, correct_test, total_test, top5_test = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, widths, heights, labels in test_loader:
                images = images.to(device)
                widths, heights = widths.to(device), heights.to(device)
                labels = labels.to(device)

                outputs = model(images, widths, heights)

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.topk(5, dim=1)
                correct_test += (predicted[:, 0] == labels).sum().item()
                total_test += labels.size(0)
                top5_test += (predicted == labels.view(-1, 1)).sum().item()

        test_losses = test_loss / len(test_loader)
        test_accuracies = correct_test / total_test
        test_top5s = top5_test / total_test

        print(f"Training Loss: {train_loss:.4f}, Test Loss: {test_losses:.4f}")
        print(f"Top-1 Accuracy: Train={train_accuracy:.4f}, Test={test_accuracies:.4f}")
        print(f"Top-5 Accuracy: Train={train_top5s:.4f}, Test={test_top5s:.4f}")

        # Logging
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_losses)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["test_accuracy"].append(test_accuracies)
        metrics["train_top5"].append(train_top5s)
        metrics["test_top5"].append(test_top5s)

        log_metrics(metrics, epoch, metrics_path)

        # Model Checkpoint
        model_path = os.path.join(model_dir, f"checkpoint_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

        # Live Plotting
        axes[0].clear()
        axes[0].plot(metrics["train_loss"], label="Train Loss")
        axes[0].plot(metrics["test_loss"], label="Test Loss")
        axes[0].legend()
        axes[0].set_title("Loss")

        axes[1].clear()
        axes[1].plot(metrics["train_accuracy"], label="Train Accuracy")
        axes[1].plot(metrics["test_accuracy"], label="Test Accuracy")
        axes[1].legend()
        axes[1].set_title("Accuracy")

        axes[2].clear()
        axes[2].plot(metrics["train_top5"], label="Train Top 5 Accuracy")
        axes[2].plot(metrics["test_top5"], label="Test Top 5 Accuracy")
        axes[2].legend()
        axes[2].set_title("Top 5 Accuracy")
        plt.savefig(os.path.join(model_dir, "plt.pdf"))

    plt.show()


# Data loading and processing
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "mixed_dataset")
    csv_path = os.path.join(base_dir, "mixed_dataset.csv")
    model_dir = os.path.join(base_dir, "augmented_models")
    metrics_path = os.path.join(model_dir, "augmented_metrics.csv")

    os.makedirs(model_dir, exist_ok=True)
    dataset = SymbolDataset(
        csv_file=csv_path,
        root_dir=dataset_dir,
        transform=transform,
        load_directly=True,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 5000
    num_workers = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=2,
    )

    num_classes = len(symbols)
    model = SymbolCNN(num_classes=num_classes, image_size=image_size)
    train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.002)
