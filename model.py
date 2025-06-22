import torch
import torchvision
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(num_classes: int, seed: int = 42):
    # Load pretrained weights and transforms
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()
    
    # Load base model
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    # Freeze all base model layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Set random seed
    torch.manual_seed(seed)

    # Update classifier head for your specific number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    ).to(device)

    return model, transforms
