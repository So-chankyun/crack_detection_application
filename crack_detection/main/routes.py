from flask import (render_template, request, Blueprint
                    , redirect, current_app,url_for)
from crack_detection.main.forms import TargetForm
from PIL import Image
import os

main = Blueprint('main',__name__)

@main.route("/",methods=['GET','POST'])
def home():
    form = TargetForm()
    # 만약 데이터가 입력되지 않은 상태라면
    if form.validate_on_submit():
        # 만약 올바르게 데이터가 제출되었고, 이미지가 제대로 들어왔다면 DB에 저장.
        # 그리고 predict로 이동
        img_file = form.crack_image.data # list로 반환되려나...? 한번 확인해봐야함.
        print(len(img_file))
        # 일단은 저장하지말고 바로 predict로 보내보자.
        if img_file:
            # 일단은 원본으로 저장하자. 어차피 전처리 단계에서 모델에 맞는 크기로 변경한다.
            # test code가 이미지의 경로로 데이터를 load하기 때문에 이렇게 해야한다.
            store_path = os.path.join(current_app.root_path,'static/img',img_file.filename)
            Image.open(img_file).save(store_path)

        return redirect(url_for('detection.predict'))
    return render_template('home.html',form=form)