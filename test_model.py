import torch
from model import MNISTNet
import pytest
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    logger.info("Starting parameter count test...")
    model = MNISTNet()
    param_count = count_parameters(model)
    logger.info(f"Total trainable parameters: {param_count:,}")
    
    # Log parameter details for each layer
    logger.info("\nParameter distribution by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.numel():,} parameters")
    
    assert param_count < 20000, f"Model has {param_count:,} parameters, should be less than 20,000"
    logger.info("Parameter count test passed successfully!")

def test_model_accuracy():
    logger.info("Starting model accuracy test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MNISTNet().to(device)
    
    # Load the latest saved model
    import glob
    model_files = glob.glob('mnist_model_acc*.pth')
    assert len(model_files) > 0, "No trained model found"
    
    latest_model = max(model_files)
    logger.info(f"Loading model from: {latest_model}")
    model.load_state_dict(torch.load(latest_model))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    logger.info(f"Test dataset size: {len(test_dataset):,} images")
    
    model.eval()
    correct = 0
    total = 0
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 2 == 0:
                logger.info(f"Processed {total:,} images...")
    
    accuracy = 100 * correct / total
    logger.info(f"\nFinal Results:")
    logger.info(f"Total images tested: {total:,}")
    logger.info(f"Correct predictions: {correct:,}")
    logger.info(f"Model accuracy: {accuracy:.2f}%")
    
    assert accuracy > 99.0, f"Model accuracy is {accuracy:.2f}%, should be > 99.0%"
    logger.info("Accuracy test passed successfully!")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 