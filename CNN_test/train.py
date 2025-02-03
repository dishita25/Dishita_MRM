import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import MNIST_CNN

def train_model():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MNIST_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 5
    loss_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_values.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
    
    # Save model
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved successfully.")

    # Plot loss vs iteration graph
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, label="Training Loss", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()