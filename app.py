from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras_cv.models import EfficientNetV2Backbone

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model(
    'model/skin_cancer_model.h5',
    custom_objects={'EfficientNetV2Backbone': EfficientNetV2Backbone}
)

def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected", 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction = predict_image(file_path)

    if prediction is not None and len(prediction) > 0:
        if len(prediction[0]) == 1:
            malignant_score = float(prediction[0][0])
            benign_score = 1 - malignant_score
        elif len(prediction[0]) >= 2:
            benign_score = float(prediction[0][0])
            malignant_score = float(prediction[0][1])
        else:
            return "Invalid prediction format", 500
    else:
        return "Prediction failed", 500

    classes = ['Benign', 'Malignant']
    result = classes[np.argmax([benign_score, malignant_score])]

    return render_template(
        'result.html',
        prediction=result,
        benign_score=benign_score,
        malignant_score=malignant_score,
        image_path=file_path
    )

if __name__ == '__main__':
    app.run(debug=True)
