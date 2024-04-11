from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Chemin pour enregistrer les fichiers temporaires
UPLOAD_FOLDER = 'temp_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# S'assurer que le dossier d'upload existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path='data/modele_couverture_livre.tflite')
interpreter.allocate_tensors()

# Obtenir les index des tenseurs d'entrée et de sortie
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Charger l'image et préparer les données d'entrée
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32')
            img_array /= 255.0  # Normaliser

            # Utiliser l'interpréteur TFLite pour faire la prédiction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            # Supprimer le fichier temporaire
            os.remove(filepath)

            # Affichage du résultat
            result = "Ce livre est prédit comme un bon livre." if prediction[0][0] > 0.5 else "Ce livre est prédit comme un mauvais livre."
            return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
