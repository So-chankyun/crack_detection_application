from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, MultipleFileField, SelectField, DecimalRangeField
from wtforms.validators import DataRequired, NumberRange

res_choices = [('qHD','960x540(qHD)'),('HD','1280x720(HD)'),('FHD','1920x1080(FHD)')]
# res_choices = [((960,540),'960x540(qHD)'),((1280,720),'1280x720(HD)'),((1920,1080),'1920x1080(FHD)')]
# frame_choices = [(15,'15fps'),(20,'20fps'),(25,'25fps'),(30,'30fps')]
# threshold_choices = [(0.5,'0.5%'),(0.7,'0.7%')]

# 입력받는 이미지
class TargetForm(FlaskForm):
    crack_video = MultipleFileField('Videos',render_kw={'multiple': True})
    resolution = SelectField('Resolution',choices=res_choices,coerce=str)
    frame = DecimalRangeField('Frame',default=15.0,validators=[NumberRange(min=15.0,max=30.0)])
    threshold = DecimalRangeField('Threshold',default=1.0,validators=[NumberRange(min=0.5,max=2.0)])

    submit = SubmitField('Predict')

