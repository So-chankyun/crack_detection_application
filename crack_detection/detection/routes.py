from flask import (render_template, url_for, flash,
                   redirect, request, abort, current_app,Blueprint)
from crack_detection.detection.forms import PredictionForm
from crack_detection.detection.model import test
from PIL import Image
import os

detection = Blueprint('detection', __name__)

@detection.route("/predict", methods=['GET','POST'])
def predict():
    form = PredictionForm()

    # 경로지정
    img_folder = os.path.join(current_app.root_path,'static/img')
    pred_folder = os.path.join(current_app.root_path,'static/pred')

    # 예측
    img = os.listdir(img_folder)[-1]
    img_name, ext = os.path.splitext(img)
    img_path = os.path.join(img_folder,img)
    pred = test.predict(img_path)

    # store pred
    pred_path = os.path.join(pred_folder,img_name+'_pred'+ext)
    print(pred_path)
    pred.save(pred_path)

    img_file = url_for('static',filename='img/'+img)
    pred_file = url_for('static',filename='pred/'+img_name+'_pred'+ext)

    return render_template('predict.html',form=form,crack_image=img_file,pred_image=pred_file)

