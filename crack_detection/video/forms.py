from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, MultipleFileField
from wtforms.validators import DataRequired

# 입력받는 이미지
class TargetForm(FlaskForm):
    crack_video = MultipleFileField('Videos',render_kw={'multiple': True})
    submit = SubmitField('Predict')

