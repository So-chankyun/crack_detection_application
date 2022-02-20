from flask import (render_template, request, Blueprint
                    , redirect, current_app,url_for, flash)
from crack_detection.image.forms import TargetForm
from PIL import Image
import os
import shutil

image = Blueprint('image',__name__)

@image.route("/image",methods=['GET','POST'])
def input():
    form = TargetForm()
    # 만약 데이터가 입력되지 않은 상태라면
    if form.validate_on_submit():
        # 만약 올바르게 데이터가 제출되었고, 이미지가 제대로 들어왔다면 DB에 저장.
        # 그리고 predict로 이동

        img_files = form.crack_image.data

        if img_files:
            # 일단은 원본으로 저장하자. 어차피 전처리 단계에서 모델에 맞는 크기로 변경한다.
            # test code가 이미지의 경로로 데이터를 load하기 때문에 이렇게 해야한다.
            APP_ROOT_PATH = current_app.root_path

            # ip 주소 저장
            client_ip = request.environ.get('HTTP_X_REAL_IP',request.remote_addr).replace('.','')
            client_folder = os.path.join(APP_ROOT_PATH,'static',client_ip)
            
            # ip주소로 이루어진 폴더가 존재하는가?
            if os.path.exists(client_folder):
                shutil.rmtree(client_folder+'/true/img',ignore_errors=True)
                shutil.rmtree(client_folder+'/pred/img',ignore_errors=True)
            else:
                os.mkdir(client_folder)
            
            os.makedirs(client_folder+'/true/img')
            os.makedirs(client_folder+'/pred/img')

            for img in img_files:
                store_path = os.path.join(client_folder,'true/img',img.filename)
                Image.open(img).save(store_path)

            flash("File successfully uploaded")

        return redirect(url_for('detection.img_predict'))
    return render_template('image.html',form=form)