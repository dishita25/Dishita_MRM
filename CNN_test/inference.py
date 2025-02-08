import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
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

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load Test Set
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Load Validation Set (20% of the original training data)
    full_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    _, valset = random_split(full_trainset, [train_size, val_size])
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Load Model
    model = load_model()

    # Test Set Evaluation
    test_predictions, test_actuals = predict(model, testloader)
    test_f1 = f1_score(test_actuals, test_predictions, average='weighted')
    test_accuracy = accuracy_score(test_actuals, test_predictions)

    print("Test Set Evaluation")
    print(f"F1 Score (weighted): {test_f1:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    plot_confusion_matrix(test_actuals, test_predictions, title="Confusion Matrix - Test Set")

    # Validation Set Evaluation
    val_predictions, val_actuals = predict(model, valloader)
    val_f1 = f1_score(val_actuals, val_predictions, average='weighted')
    val_accuracy = accuracy_score(val_actuals, val_predictions)

    print("\nValidation Set Evaluation")
    print(f"F1 Score (weighted): {val_f1:.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    plot_confusion_matrix(val_actuals, val_predictions, title="Confusion Matrix - Validation Set")