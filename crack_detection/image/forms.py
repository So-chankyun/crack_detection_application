from wsgiref.validate import validator
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, MultipleFileField, SelectField
from wtforms.validators import DataRequired

choices = [('qHD','960x540(qHD)'),('HD','1280x720(HD)'),('FHD','1920x1080(FHD)')]

# 입력받는 이미지
class TargetForm(FlaskForm):
    crack_image = MultipleFileField('Images',render_kw={'multiple': True})
    resolution = SelectField('Resolution',choices=choices,coerce=str)
    submit = SubmitField('Predict')

