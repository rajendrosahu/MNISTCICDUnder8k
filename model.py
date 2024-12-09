import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First Layer: 1->16 channels (increased from 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05)  # Reduced dropout
        )
        
        # Second Layer: 16->24 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Third Layer: 24->32 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step at each batch
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{100.0 * correct/processed:.2f}%'
        })
    
    return 100.0 * correct/processed

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # CUDA setup
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.1)  # Added random erasing
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    # Reduced batch size for better generalization
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # Model setup
    model = MNISTNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {param_count}')
    
    # Modified optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Warm up for 20% of training
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000  # Final lr = initial_lr/1000
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

 