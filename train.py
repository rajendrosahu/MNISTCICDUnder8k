from model import MNISTNet, train_epoch, test
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.1)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model
    model = MNISTNet().to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(1, 21):
        train_acc = train_epoch(model, device, train_loader, optimizer, scheduler, epoch)
        test_acc = test(model, device, test_loader)
        
        if test_acc >= 99.0 and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'mnist_model_acc{test_acc:.2f}.pth')
            logger.info(f'Model saved with accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main() 