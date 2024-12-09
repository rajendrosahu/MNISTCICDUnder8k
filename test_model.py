import torch
from model import MNISTNet
import pytest
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    
    # Load the latest saved model
    import glob
    model_files = glob.glob('mnist_model_acc*.pth')
    assert len(model_files) > 0, "No trained model found"
    
    latest_model = max(model_files)
    model.load_state_dict(torch.load(latest_model))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 99.0, f"Model accuracy is {accuracy:.2f}%, should be > 99.0%"

if __name__ == "__main__":
    pytest.main([__file__]) 