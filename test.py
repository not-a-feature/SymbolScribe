import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt

from train import SymbolCNN, SymbolDataset, image_size
from symbols import symbols
import numpy as np

top_N = 5
num_checkpoints = 50
num_chunks = 5
skip_chunk = 1


def validate(model, dataloader, num_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    k_values = range(1, top_N + 1)
    topk_correct = {k: 0 for k in k_values}

    with torch.no_grad():
        for images, widths, heights, labels in dataloader:
            images = images.to(device)
            widths, heights = widths.to(device), heights.to(device)
            labels = labels.to(device)

            output = model(images, widths, heights)

            _, predicted = output.topk(max(k_values), 1, True, True)

            for k in k_values:
                topk_correct[k] += (predicted[:, :k] == labels.unsqueeze(1)).sum().item()

    topk_acc = {k: correct / num_samples for k, correct in topk_correct.items()}
    return topk_acc


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "test_dataset")
    csv_path = os.path.join(base_dir, "test_dataset.csv")

    model_dir = os.path.join(base_dir, "augmented_models_3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = SymbolDataset(
        csv_file=csv_path,
        root_dir=output_dir,
        transform=transform,
        crop_to_content=True,
        load_directly=True,
    )
    num_samples = len(dataset)
    batch_size = min(num_samples, 10000)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    plt.figure(facecolor="#faf8f8")

    best_acc = 0
    best_checkpoint = ""
    best_accuracies = None
    all_accuracies = []

    for i in range(0, num_checkpoints):
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{i}.pth")  # Corrected f-string

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found. Skipping.")
            continue

        model = SymbolCNN(num_classes=len(symbols))

        cpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(cpt)
        model.eval()
        model.to(device)

        topk_acc = validate(model, dataloader, num_samples=num_samples)

        k_values = list(topk_acc.keys())
        accuracies = list(topk_acc.values())
        all_accuracies.append(accuracies)

        avg_accuracie = sum(accuracies) / len(accuracies)
        if avg_accuracie > best_acc:
            best_acc = avg_accuracie
            best_checkpoint = i
            best_accuracies = accuracies

    all_accuracies = np.array(all_accuracies)
    chunk_size = num_checkpoints // num_chunks

    for j in range(skip_chunk, num_chunks):
        start = j * chunk_size
        end = (j + 1) * chunk_size
        chunk_accuracies = all_accuracies[start:end]
        avg_acc = np.mean(chunk_accuracies, axis=0)

        plt.plot(
            k_values,
            avg_acc,
            alpha=0.3,
            label=f"Avg. Epoch {start} - {end}",
        )

    plt.plot(
        k_values,
        best_accuracies,
        marker="o",
        color="#4caf50",
        label=f"Best Checkpoint (Epoch {best_checkpoint})",
    )
    print(accuracies)

    plt.xlabel("k")
    plt.ylabel("Top-k Accuracy")
    plt.title("Top-k Accuracy vs. k for Different Checkpoints")
    plt.grid(True)
    plt.legend()

    print(f"Best checkpoint: {best_checkpoint} with accuracy {best_acc}")
    plt.show()
