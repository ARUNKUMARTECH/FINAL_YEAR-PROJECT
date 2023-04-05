from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
from io import BytesIO


app = Flask(__name__)

labels = ['desert', 'mountains', 'sea', 'sunset', 'trees']

model = load_model('datamodel.h5')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)
    processed_image = preprocess_image(image, target_size=(64, 64))
    predicted_labels = model.predict(processed_image)[0]
    label_prob_pairs = {label: prob for label, prob in zip(labels, predicted_labels)}
    sorted_label_prob_pairs = sorted(label_prob_pairs.items(), key=lambda x: x[1], reverse=True)
    sorted_labels = [(label, round(prob * 100, 2)) for label, prob in sorted_label_prob_pairs]

    buffered = BytesIO()
    Image.fromarray(np.uint8(processed_image[0])).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('prediction.html', sorted_labels=sorted_labels, img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
