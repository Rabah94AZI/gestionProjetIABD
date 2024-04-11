import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modèle enregistré
model = tf.keras.models.load_model('modele_couverture_livre.h5')

# Charger une image de test (remplacez 'chemin_vers_votre_image.jpg' par le chemin de votre image)
image_path = 'C:/Users/abdou/Desktop/image/test.jpg'
img = image.load_img(image_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Convertir en lot unique
img_array /= 255.0  # Normaliser l'image

# Faire une prédiction
prediction = model.predict(img_array)

# Afficher le résultat
if prediction[0][0] > 0.5:
    print("Ce livre est prédit comme un bon livre.")
else:
    print("Ce livre est prédit comme un mauvais livre.")
