from flask import (render_template, url_for, flash,
                   redirect, request, abort, current_app,Blueprint, send_file)
from crack_detection.detection.forms import PredictedImageForm, PredictedVideoForm
from crack_detection.detection.model import test
from crack_detection.detection.utils import predict_video
from PIL import Image
import os
from tqdm import tqdm
from collections import namedtuple
import shutil
import torchvision.transforms as tr

def get_client_info():
    APP_ROOT_PATH = current_app.root_path

    # ip 주소 저장
    client_ip = request.environ.get('HTTP_X_REAL_IP',request.remote_addr).replace('.','')
    client_folder = os.path.join(APP_ROOT_PATH,'static',client_ip)

    return client_ip, client_folder

detection = Blueprint('detection', __name__)

@detection.route("/img_predict", methods=['GET','POST'])
def img_predict():

    client_ip, client_folder = get_client_info()
    
    # 경로지정
    img_folder = os.path.join(client_folder,'true/img')
    pred_folder = os.path.join(client_folder,'pred/img')

    # 예측
    print('predict')
    img_list = os.listdir(img_folder)
    print(img_list)
    ext = os.path.splitext(img_list[0])[1]
    img_name_list = [os.path.splitext(file_name)[0]  for file_name in img_list]
    img_path_list = [os.path.join(img_folder,file_name) for file_name in img_list]

    # load model
    model = test.load_model()
    pred = [tr.ToPILImage(test.predict(path,'image',model)) for path in tqdm(img_path_list)]

    print('store')
    # store pred
    pred_path = []
    for i in range(len(img_list)):
        pred_path.append(os.path.join(pred_folder,img_name_list[i]+'_pred'+ext))
        pred[i].save(pred_path[i])
        print(pred_path[i])

    # url_for 객체를 저장한다.
    result = namedtuple('result',['crack_image','pred_image'])
    result_url = []

    for name in img_name_list:
        # img_file_name = os.path.join(client_ip,'true/img/'+name+ext)
        # pred_file_name = os.path.join(client_ip,'pred/img/'+name+'_pred'+ext)
        # print('img_file_name : {}'.format(img_file_name))

        img_file_url = url_for('static',filename=client_ip+'/true/img/'+name+ext)
        pred_file_url = url_for('static',filename=client_ip+'/pred/img/'+name+'_pred'+ext)
        print(img_file_url)
        result_url.append(result(img_file_url,pred_file_url))

    data = {'result_list':result_url}

    form = PredictedImageForm(data=data)

    return render_template('img_predict.html',form=form)

@detection.route("/download")
def download():
    # db에서 유저를 구별하여 해당 폴더를 다운로드 할수 있도록 해야한다.
    # db구성을 어떻게 해야하나...
    # 일단은 해당 폴더를 지정하여 저장이되는지 한번 살펴보자

    client_ip, client_folder = get_client_info()
    
    # zip file 저장
    zip_store_path = client_folder+'_result'
    shutil.make_archive(zip_store_path,'zip',client_folder)  # 저장하고자하는 이름, 확장자, 해당 폴더

    return send_file(zip_store_path+'.zip',
                    attachment_filename="pred.zip",
                    as_attachment=True)

@detection.route("/video_predict", methods=['GET','POST'])
def video_predict():

    client_ip, client_folder = get_client_info()

    # 경로지정
    video_folder = os.path.join(client_folder,'true/video')
    pred_folder = os.path.join(client_folder,'pred/video')

    # 예측
    print('predict')
    video_list = os.listdir(video_folder)
    ext = os.path.splitext(video_list[0])[1]
    video_name_list = [os.path.splitext(file_name)[0] for file_name in video_list]
    video_path_list = [os.path.join(video_folder,file_name) for file_name in video_list]

    result = namedtuple('result',['crack_video','pred_video'])
    result_url = []

    print('store')
    # store pred
    pred_path = []
    for i, video_name in tqdm(enumerate(video_name_list)):
        pred_path = os.path.join(pred_folder,video_name+'_pred'+ext)
        predict_video.make_video(video_path_list[i],pred_path,pred_folder) # 변환하고 바로 저장.

        # url_for 객체를 저장한다.
        video_file = url_for('static',filename=client_ip+'/true/video/'+video_name+ext)
        pred_file = url_for('static',filename=client_ip+'/pred/video/'+video_name+'_pred'+ext)
        result_url.append(result(video_file,pred_file))

    data = {'result_list':result_url}

    form = PredictedVideoForm(data=data)

    return render_template('video_predict.html',form=form)