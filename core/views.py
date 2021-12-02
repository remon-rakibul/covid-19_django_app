import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage






# Create your views here.


def home(request):
    return render(request, 'core/index.html')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds


# Load model
from tensorflow.keras.models import load_model
MODEL_PATH = './ml_model/covid19.model'
model = load_model(MODEL_PATH)

def predict(request):
    if request.method == 'POST':
        # Get the file from post request
        f = request.FILES['file']

        # Save the file
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        img_path = 'media/{}'.format(file_name_2)

        # Make prediction
        preds = model_predict(img_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)  # Simple argmax
        result = str(pred_class)  # Convert to string
        if result == '[1]':
            return render(request, 'core/result.html', {'img': file_name_2, 'response': "Negative - Patient Normal", 'advice': 'stay safe'})
        else:
            return render(request, 'core/result.html', {'img': file_name_2, 'response': "Positive - covid-19 affected", 'advice': 'Consult to a Doctor ASAP'})
    else:
        return render(request, 'core/index.html')
