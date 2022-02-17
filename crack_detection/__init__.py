from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from crack_detection.config import Config

db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    from crack_detection.main.routes import main
    from crack_detection.image.routes import image
    from crack_detection.video.routes import video
    from crack_detection.detection.routes import detection
    from crack_detection.errors.handlers import errors

    app.register_blueprint(main)
    app.register_blueprint(errors)
    app.register_blueprint(image)
    app.register_blueprint(video)
    app.register_blueprint(detection)

    return app