from flask import (render_template, url_for, flash,
                   redirect, request, abort, current_app,Blueprint)
from crack_detection.detection.forms import PredictedImageForm, PredictedVideoForm
from crack_detection.detection.model import test
from crack_detection.detection.utils import predict_video
from PIL import Image
import os
from tqdm import tqdm
from collections import namedtuple

detection = Blueprint('detection', __name__)

@detection.route("/img_predict", methods=['GET','POST'])
def img_predict():
    # 경로지정
    img_folder = os.path.join(current_app.root_path,'static/true/img')
    pred_folder = os.path.join(current_app.root_path,'static/pred/img')

    # 예측
    print('predict')
    img_list = os.listdir(img_folder)
    ext = os.path.splitext(img_list[0])[1]
    img_name_list = [os.path.splitext(file_name)[0]  for file_name in img_list]
    img_path_list = [os.path.join(img_folder,file_name) for file_name in img_list]

    pred = [test.predict(path,'image') for path in tqdm(img_path_list)]

    print('store')
    # store pred
    pred_path = []
    for i in range(len(img_list)):
        pred_path.append(os.path.join(pred_folder,img_name_list[i]+'_pred'+ext))
        print(pred)
        pred[i].save(pred_path[i])

    # url_for 객체를 저장한다.
    result = namedtuple('result',['crack_image','pred_image'])
    result_url = []

    for name in img_name_list:
        img_file = url_for('static',filename='/true/img/'+name+ext)
        pred_file = url_for('static',filename='/pred/img/'+name+'_pred'+ext)
        result_url.append(result(img_file,pred_file))

    data = {'result_list':result_url}

    form = PredictedImageForm(data=data)

    return render_template('img_predict.html',form=form)

@detection.route("/video_predict", methods=['GET','POST'])
def video_predict():
    # 경로지정
    video_folder = os.path.join(current_app.root_path,'static/true/video')
    pred_folder = os.path.join(current_app.root_path,'static/pred/video')

    # 예측
    print('predict')
    video_list = os.listdir(video_folder)
    print()
    ext = os.path.splitext(video_list[0])[1]
    video_name_list = [os.path.splitext(file_name)[0]  for file_name in video_list]
    video_path_list = [os.path.join(video_folder,file_name) for file_name in video_list]

    result = namedtuple('result',['crack_video','pred_video'])
    result_url = []

    print('store')
    # store pred
    pred_path = []
    for i, video_name in tqdm(enumerate(video_name_list)):
        pred_path = os.path.join(pred_folder,video_name+'_pred'+ext)
        predict_video.make_video(video_path_list[i],pred_path) # 변환하고 바로 저장.

        # url_for 객체를 저장한다.
        video_file = url_for('static',filename='/true/video/'+video_name+ext)
        pred_file = url_for('static',filename='/pred/video/'+video_name+'_pred'+ext)
        result_url.append(result(video_file,pred_file))

    data = {'result_list':result_url}

    form = PredictedVideoForm(data=data)

    return render_template('video_predict.html',form=form)