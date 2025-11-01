from modeling_siglip import *
from PIL import Image
import torch
from torchvision import transforms


def load_image_pil(path, size=224, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((size, size)),   # or Resize(size) + CenterCrop(size)
        transforms.ToTensor(),            # converts to [C,H,W], floats in [0,1]
        # optional: normalize for ImageNet pre-trained models:
        # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert('RGB')  # ensure 3 channels
    tensor = transform(img)                # shape: [3, 224, 224]
    tensor = tensor.unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
    return tensor

# usage
x = load_image_pil("Input_img.png", size=224, device='cpu')
print(x.shape)  # torch.Size([1, 3, 224, 224])

config = SiglipVisionConfig()
model = SiglipVisionModel(config)
model(x)