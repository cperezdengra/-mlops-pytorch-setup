import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.train import SimpleNN

def main():
    transform = transforms.ToTensor()
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=False, download=True, transform=transform),
        batch_size=32, shuffle=False
    )

    model = SimpleNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    print(f"📊 Accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    main()
