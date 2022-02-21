import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from crack_detection.detection.model.network.unet_model import UNet
from crack_detection.detection.model import test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

# MODEL PATH
detection_folder = './crack_detection/detection'
MODEL_PATH = os.path.join(detection_folder,'Unet(50k using data, epoch15).pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def merge_img(frame, pred,size):
    re_frm = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
    add_mask = re_frm.copy()
    add_mask[pred[:,:] !=0 ]=[0,255,0]


    return add_mask

'''
* Description *
  - 저장된 비디오를 예측하여 다시 저장한다.

* Process *
  1. Load model
  2. Load Data
  3. Predict
  4. circulate 2~3 process
'''
def make_video(video_path,save_path,pred_folder,size=(960,540),frame=15.0,threshold=1.0):
    width, height = size
    frame_thred = threshold
    frame = frame

    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,width) # 일단은 고정된 사이즈로 가자
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_path, fourcc, frame, (int(width), int(height)))

    count = 0
    max_ratio, avg_ratio = 0.0, 0.0

    model = test.load_model()

    while cv2.waitKey(33) < 0:
        count+=1
        full_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = capture.read()
        if not ret:
            print("프레임을 수신할 수 없습니다. 종료 중 ...")
            break

        pred_img = np.asarray(test.predict(frame,'video',model,size))
        convert_img = merge_img(frame, pred_img,size)
        CRPF = np.round((pred_img.reshape(-1).sum()/(size[0]*size[1]))*100, 2)
        max_ratio = CRPF if CRPF > max_ratio else max_ratio
        avg_ratio = (avg_ratio*(count-1)+CRPF)/count

        """
        ########### args.frame_thred 이상이 됐을 때 캡처 (추가 예정) ###########
        """

        font=cv2.FONT_HERSHEY_SIMPLEX
        color = (50,50,165) if CRPF > frame_thred else (50,165,50)
        
        cv2.putText(convert_img, 'Crt Rae : {:.2f} {}'.format(CRPF, "%"), (5, 20), font, .6, color, 2)
        store_img = convert_img.copy()
        cv2.putText(convert_img, 'Max Rate : {:.2f} {}'.format(max_ratio, "%"), (5, 40), font, .6, (60,180,255), 2)
        cv2.putText(convert_img, 'Avg Rate : {:.2f} {}'.format(avg_ratio, "%"), (5, 60), font, .6, (60,180,255), 2)
        cv2.putText(convert_img, 
                    'Progress Rate : {:.1f} {}'.format((count/full_frame)*100, "%"), 
                    (5, 80), font, .6, (60,180,255), 2)

        if int((count/full_frame)*100) % 10:
          print(f'진행률 : {(count / full_frame) * 100:.2f}%')
        
        # args.crack_thred 이상이 됐을 때 캡처
        if CRPF > frame_thred:
            print(f'current crack ratio : {CRPF}%')
            outfile = os.path.join(pred_folder,'capture',f"{count}.jpg")
            cv2.imwrite(outfile, store_img)

        out.write(convert_img)


    capture.release()
    out.release()