from flask import (render_template, url_for, flash,
                   redirect, request, abort, Blueprint)
from crack_detection.detection.forms import PredictionForm
from crack_detection.detection.model.test import predict
from PIL import Image

detection = Blueprint('detection', __name__)

@detection.route("/predict", methods=['GET','POST'])
def predict():
    return render_template('predict.html')

