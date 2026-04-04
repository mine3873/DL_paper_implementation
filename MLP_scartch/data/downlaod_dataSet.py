from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transforms)


