from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FloatField, StringField, FieldList, FormField
from wtforms.validators import DataRequired


# 출력하는 이미지
class ImageForm(FlaskForm):
    crack_image = StringField('Image',validators=[DataRequired()])
    pred_image = StringField('Prediction',validators=[DataRequired()])

class VideoForm(FlaskForm):
    crack_video = StringField('Image',validators=[DataRequired()])
    pred_video = StringField('Prediction',validators=[DataRequired()])

class PredictedImageForm(FlaskForm):
    result_list = FieldList(FormField(ImageForm),min_entries=1)

class PredictedVideoForm(FlaskForm):
    result_list = FieldList(FormField(VideoForm),min_entries=1)
