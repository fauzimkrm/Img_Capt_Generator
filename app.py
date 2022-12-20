import pickle
import numpy as np
import sys
import io
import PIL.Image

from PIL import Image
from googletrans import Translator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model, load_model
from flask import Flask, request, jsonify

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

def extract_features(image, model):
    # try:
    #     image = Image.open(filename)
    #
    # except:
    #     print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299, 299))
    image = np.array(image)

    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word

        if word == 'end':
            break
    return in_text

# nltk.download("stopwords")

with open('imgIdss.pkl', 'rb') as f:
    imgIdss = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('dataset_dict.pkl', 'rb') as f:
    dataset = pickle.load(f)

max_length = max_length(dataset)
model = load_model('model_5.h5')
xception_model = Xception(include_top=False, pooling="avg")
translator = Translator()
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    image_file = request.files['file']
    image_bytes = image_file.read()
    img = PIL.Image.open(io.BytesIO(image_bytes))
    photo = extract_features(img, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)
    description = translator.translate(description[6:-4], dest='id')
    result = {'description': description.text}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')