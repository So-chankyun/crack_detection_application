from flask import (render_template, request, Blueprint
                    , redirect, current_app,url_for, flash)
from crack_detection.video.forms import TargetForm
from PIL import Image
import os
import cv2
import shutil

video = Blueprint('video',__name__)

@video.route("/video",methods=['GET','POST'])
def input():
    form = TargetForm()
    # 만약 데이터가 입력되지 않은 상태라면
    if form.validate_on_submit():
        # 만약 올바르게 데이터가 제출되었고, 이미지가 제대로 들어왔다면 DB에 저장.
        # 그리고 predict로 이동

        video_files = form.crack_video.data
        resolution = form.resolution.data
        frame = form.frame.data
        threshold = form.threshold.data

        # 일단은 저장하지말고 바로 predict로 보내보자.
        if video_files:
            # 일단은 원본으로 저장하자. 어차피 전처리 단계에서 모델에 맞는 크기로 변경한다.
            # test code가 이미지의 경로로 데이터를 load하기 때문에 이렇게 해야한다.
            APP_ROOT_PATH = current_app.root_path

            # ip 주소 저장
            client_ip = request.environ.get('HTTP_X_REAL_IP',request.remote_addr).replace('.','')
            client_folder = os.path.join(APP_ROOT_PATH,'static',client_ip)
            
            # ip주소로 이루어진 폴더가 존재하는가?
            if os.path.exists(client_folder):
                shutil.rmtree(client_folder+'/true/video',ignore_errors=True)
                shutil.rmtree(client_folder+'/pred/video',ignore_errors=True)
            else:
                os.mkdir(client_folder)
                
            os.makedirs(client_folder+'/true/video')
            os.makedirs(client_folder+'/pred/video')
            os.makedirs(client_folder+'/pred/video/capture')
            
            for video in video_files:
                store_path = os.path.join(client_folder,'true/video',video.filename)
                video.save(store_path)

            flash("File successfully uploaded")
            
            print(frame, threshold)
            print(type(frame), type(threshold))

            data = {"resolution":resolution,
                    "frame":float(frame),
                    "threshold":float(threshold)}
 
        return redirect(url_for('detection.video_predict',data=data))
    return render_template('video.html',form=form)