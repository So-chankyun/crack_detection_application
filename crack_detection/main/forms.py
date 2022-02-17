from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

# 입력받는 이미지
class TargetForm(FlaskForm):
    crack_image = FileField('Image',validators=[FileAllowed(['jpg','png'])])
    submit = SubmitField('Predict')

