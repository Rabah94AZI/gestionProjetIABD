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

# Charger le modèle
model = tf.keras.models.load_model('data/modele_couverture_livre.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Récupérer l'image du formulaire
        file = request.files['file']
        if file:
            # Sécuriser le nom du fichier et le sauvegarder temporairement
            filename = secure_filename(file.filename)
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Charger l'image depuis le fichier temporaire
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normaliser

            # laprédiction
            prediction = model.predict(img_array)

            # Supprimer le fichier temporaire
            os.remove(filepath)

            #affichage
            result = "Ce livre est prédit comme un bon livre." if prediction[0][0] > 0.5 else "Ce livre est prédit comme un mauvais livre."

            # Renvoyer la page result
            return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
