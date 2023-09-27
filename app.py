# app.py
import threading

from flask import Flask, render_template, redirect, url_for
from flask_caching import Cache

from views.view import view as views_view
from views.cars import cars as views_cars

from recognizer.cars import Vehicle

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


def load_tf_model(model_path=None):
    # load_model
    model_path = './saved_models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8'
    app.config['MODEL_Vehicle'] = Vehicle()
    app.config['MODEL_Vehicle'].load_model(model_path)


load_tf_model()


@app.route('/')
def index():
    return redirect(url_for('view.status'))


app.register_blueprint(views_view, url_prefix='/view')
app.register_blueprint(views_cars, url_prefix='/cars')

if __name__ == '__main__':
    app.run()
