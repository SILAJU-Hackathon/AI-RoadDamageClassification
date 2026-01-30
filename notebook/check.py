from pathlib import Path
from random import shuffle
from utils.eval import compare
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    CURRENT_DIR = Path(__file__).resolve().parent
    DATA_DIR = CURRENT_DIR.parent / "data" / "processed" / "test"

    tf = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])

    image_datasets = datasets.ImageFolder(DATA_DIR, tf)
    test_loader = DataLoader(image_datasets,batch_size=8,shuffle=False,num_workers=2)
    compare(CURRENT_DIR.parent / "notebook" / "save_model", test_loader,None)
    
if __name__ == '__main__':
    main()