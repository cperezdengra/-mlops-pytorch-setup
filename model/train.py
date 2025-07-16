import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 28*28))

def main():
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True, transform=transform),
        batch_size=32, shuffle=True
    )

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} – Loss: {avg_loss:.4f}")

        # Guardar el loss de la segunda epoch para comprobarlo en CI
        if epoch == 1:
            with open("second_epoch_loss.txt", "w") as f:
                f.write(f"{avg_loss:.4f}")
        #print(f"Epoch {epoch+1} – Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    main()
