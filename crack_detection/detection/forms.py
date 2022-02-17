from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FloatField, StringField, FieldList, FormField
from wtforms.validators import DataRequired


# 출력하는 이미지
class DataForm(FlaskForm):
    crack_image = StringField('Image',validators=[DataRequired()])
    pred_image = StringField('Prediction',validators=[DataRequired()])

class PredictionForm(FlaskForm):
    result_list = FieldList(FormField(DataForm),min_entries=1)
