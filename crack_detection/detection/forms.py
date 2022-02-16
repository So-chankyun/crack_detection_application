from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FloatField, StringField
from wtforms.validators import DataRequired


# 출력하는 이미지
class PredictionForm(FlaskForm):
    # mask_ratio = FloatField('Mask Ratio',validators=[DataRequired()])
    # crack_image_path = StringField('Image',validators=[DataRequired()])
    crack_image = FileField('Image',validators=[FileAllowed(['png','jpg'])])
    pred_image = FileField('Prediction',validators=[FileAllowed(['png','jpg'])])
