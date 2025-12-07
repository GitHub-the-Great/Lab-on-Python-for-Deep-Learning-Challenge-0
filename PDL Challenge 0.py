import torcport torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

def get_cifar10_loader(N, batch_size=128):
    """ Load CIFAR-10 dataset with custom class distribution."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        indices[label].append(idx)
    
    selected_indices = []
    for i, n in enumerate(N):
        selected_indices.extend(indices[i][:n])
    
    train_subset = Subset(dataset, selected_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    """ Train ResNet18 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_acc = []
    for epoch in range(num_epochs):
        correct, total = 0, 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        train_acc.append(acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {acc:.2f}%')
    
    return train_acc

def evaluate_model(model, test_loader):
    """ Evaluate model accuracy per class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    per_class_accuracy = {i: 100 * class_correct[i] / class_total[i] for i in range(10)}
    return per_class_accuracy

def plot_results(train_acc, per_class_accuracy):
    """ Plot training accuracy and per-class accuracy."""
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, marker='o', label='Training Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy Over Epochs")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(per_class_accuracy.keys()), y=list(per_class_accuracy.values()))
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Load and preprocess dataset
    N = [4000] * 10  # Balanced dataset
    train_loader, test_loader = get_cifar10_loader(N)
    
    # Step 2: Define ResNet18
    model = resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Step 3: Train model
    train_acc = train_model(model, train_loader, criterion, optimizer, num_epochs=20)
    
    # Step 4: Evaluate model
    per_class_accuracy = evaluate_model(model, test_loader)
    
    # Step 6: Plot results
    plot_results(train_acc, per_class_accuracy)
    
    # Step 5: Investigate dataset imbalance
    N_imbalanced = [6000 if i == 0 else 3777 for i in range(10)]
    train_loader_imbal, _ = get_cifar10_loader(N_imbalanced)
    
    model = resnet18(num_classes=10)
    train_acc_imbal = train_model(model, train_loader_imbal, criterion, optimizer, num_epochs=20)
    per_class_accuracy_imbal = evaluate_model(model, test_loader)
    
    plot_results(train_acc_imbal, per_class_accuracy_imbal)
    
    # Step 7: Modify loss function to mitigate imbalance
    class_weights = torch.tensor([1/6000 if i == 0 else 1/3777 for i in range(10)], dtype=torch.float32).to(device)
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = resnet18(num_classes=10)
    train_acc_weighted = train_model(model, train_loader_imbal, weighted_criterion, optimizer, num_epochs=20)
    per_class_accuracy_weighted = evaluate_model(model, test_loader)
    
    plot_results(train_acc_weighted, per_class_accuracy_weighted)

if __name__ == "__main__":
    main()
