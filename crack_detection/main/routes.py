from flask import (render_template, request, Blueprint
                    , redirect, current_app,url_for, flash)
from crack_detection.image.forms import TargetForm
from PIL import Image
import os

main = Blueprint('main',__name__)

# home 화면
@main.route("/")
def home():
    return render_template('home.html')