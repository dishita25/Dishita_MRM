import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import MNIST_CNN
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path="mnist_cnn.pth"):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, data_loader):
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            actuals.extend(labels.tolist())
    return predictions, actuals

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    model = load_model()
    predictions, actuals = predict(model, testloader)
    
    # Compute Metrics
    f1 = f1_score(actuals, predictions, average='weighted')
    accuracy = accuracy_score(actuals, predictions)  
    
    # Display first 10 predictions, actual labels, and accuracy
    print("Sample Predictions:", predictions[:10])
    print("Actual Labels:", actuals[:10])
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")  
    
    # Plot confusion matrix
    plot_confusion_matrix(actuals, predictions)