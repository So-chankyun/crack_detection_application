from flask import (render_template, request, Blueprint
                    , redirect, current_app,url_for)
from crack_detection.main.forms import TargetForm
from crack_detection.detection.forms import PredictionForm
from crack_detection.detection.model.test import predict
from PIL import Image
import os

main = Blueprint('main',__name__)

@main.route("/")
@main.route("/home",methods=['GET','POST'])
def home():
    form = TargetForm()
    # 만약 데이터가 입력되지 않은 상태라면
    if form.validate_on_submit():
        # 만약 올바르게 데이터가 제출되었고, 이미지가 제대로 들어왔다면 DB에 저장.
        # 그리고 predict로 이동
        form = PredictionForm()
        img_file = form.crack_image.data

        # 일단은 저장하지말고 바로 predict로 보내보자.
        # if img_file:
        #     form = PredictionForm()
        #     # 일단은 원본으로 저장하자. 어차피 전처리 단계에서 모델에 맞는 크기로 변경한다.
        #     # test code가 이미지의 경로로 데이터를 load하기 때문에 이렇게 해야한다.
        #     store_path = os.path.join(current_app.root_path,'static/img',img_file.filename)
        #     Image.open(img_file).save(store_path)

        # Image.open()이 이미지 경로도 되고 raw data도 되는지 확인해보자
        pred_img = predict(img_file)

        return render_template('prediction.html',form=form, crack_image=img_file,pred_image=pred_img)
    return render_template('home.html',form=form)