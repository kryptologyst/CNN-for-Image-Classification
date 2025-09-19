import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'best_model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    if model is None:
        return "Model not found. Please train the model first by running main.py"
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(file_path)

            preds = model_predict(file_path, model)
            pred_class = CLASS_NAMES[np.argmax(preds)]

            return pred_class
    return None

if __name__ == '__main__':
    app.run(debug=True)
