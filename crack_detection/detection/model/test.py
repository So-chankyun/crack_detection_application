import os
from PIL import Image
from crack_detection.detection.model.network.unet_model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

detection_folder = './crack_detection/detection'
MODEL_PATH = os.path.join(detection_folder,'Unet(50k using data, epoch15).pth')
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

    # transform image
    tensor_img = tr.ToTensor()(img).unsqueeze(0)
    tensor_img.to(DEVICE)

    return tensor_img

# 3. prediction
def predict(img_path):
    model = load_model()
    pred = model(load_data(img_path))
    pred = torch.argmax(F.softmax(pred,dim=1),dim=1).float()
    convert_img = tr.ToPILImage()(pred)

    return convert_img









