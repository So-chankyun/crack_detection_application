from datetime import datetime
from crack_detection import db

class TargetImage(db.Model):
    # table name
    __tablename__ = 'targetImage'

    # attribute
    id = db.Column(db.Integer, primary_key=True)
    target_img_file = db.Column(db.String(50), nullable=False, default='default.jpg')

class PredictImage(db.Model):
    '''
    * PRIMARY KEY *
    target_id : 예측하고자하는 원본 이미지의 id이다. foreign key이다.
    target_img_file : 예측하고자하는 이미지의 원본 링크를 나타낸다. foreign key이다.

    * ATTRIBUTE *
    pred_img_file : 예측된 크랙이미지의 링크를 나타낸다. null을 가질수 없다.

    '''
    # table name
    __tablename__ = 'predImage'

    # attribute
    target_id = db.Column(db.Integer, db.ForeignKey('targetImage.id'), nullable=False, primary_key=True)
    # targetImage table의 id를 참조함. 그리고 primary key로 사용한다.
    target_img_file = db.Column(db.String(50),db.ForeignKey(TargetImage.target_img_file),nullable=False, primary_key=True)
    pred_img_file = db.Column(db.String(50),nullable=False,default='default_pred.jpg')
    # mask_ratio = db.Column(db.Float, nullable=False)
    # 이건 나중에 추가하도록 하자. 일단 predict해서 화면에 띄우는 것 까지만 진행