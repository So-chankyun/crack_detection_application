from flask import (render_template, url_for, flash,
                   redirect, request, abort, current_app,Blueprint)
from crack_detection.detection.forms import PredictionForm
from crack_detection.detection.model import test
from PIL import Image
import os
from tqdm import tqdm
from collections import namedtuple

detection = Blueprint('detection', __name__)

@detection.route("/predict", methods=['GET','POST'])
def predict():
    # 경로지정
    img_folder = os.path.join(current_app.root_path,'static/img')
    pred_folder = os.path.join(current_app.root_path,'static/pred')

    # 예측
    print('predict')
    img_list = os.listdir(img_folder)
    ext = os.path.splitext(img_list[0])[1]
    img_name_list = [os.path.splitext(file_name)[0]  for file_name in img_list]
    img_path_list = [os.path.join(img_folder,file_name) for file_name in img_list]

    pred = [test.predict(path) for path in tqdm(img_path_list)]

    print('store')
    # store pred
    pred_path = []
    for i in range(len(img_list)):
        pred_path.append(os.path.join(pred_folder,img_name_list[i]+'_pred'+ext))
        pred[i].save(pred_path[i])

    # url_for 객체를 저장한다.
    result = namedtuple('result',['crack_image','pred_image'])
    result_url = []

    for name in img_name_list:
        img_file = url_for('static',filename='img/'+name+ext)
        pred_file = url_for('static',filename='pred/'+name+'_pred'+ext)
        result_url.append(result(img_file,pred_file))

    data = {'result_list':result_url}
    print(data['result_list'])

    form = PredictionForm(data=data)

    for field in form.result_list:
        print(field.crack_image.data)
        print(field.pred_image.data)

    return render_template('predict.html',form=form)

