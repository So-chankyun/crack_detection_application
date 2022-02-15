import os
from PIL import Image
from network.unet_model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

MODEL_PATH = './Unet(50k using data).pth'
DEVICE = 'cpu' # 나중에 gpu를 사용할 수 있으면 수정하도록 하자.

# 1. Load model
def load_model():    
    model = UNet(n_channels=3,n_classes=2,bilinear=True)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    return model

# 2. Data load and transforms
def load_data(img_path):
    # load image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((640,384))

    # transform image
    tensor_img = tr.ToTensor()(img).unsqueeze(0)
    temsor_img.to(DEVICE)

    return tensor_img

# 3. prediction
def predict(model,img):
    pred = model(img)
    pred = torch.argmax(F.softmax(pred,dim=1),dim=1).float()
    convert_img = tr.ToPILImage()(pred)

    return convert_img




    




