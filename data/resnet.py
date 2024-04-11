import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

# Charger les données
print("------ Chargement des données ------")
data = pd.read_csv('C:/Users/abdou/Desktop/gp/data/main_dataset.csv')
print(data.head())

# Ajustement des chemins d'accès pour qu'ils soient complets et absolus
data['img_paths'] = data['img_paths'].apply(lambda x: os.path.join('C:/Users/abdou/Desktop/gp/data', x))

# Vérifie l'existence des fichiers d'images
data['exists'] = data['img_paths'].apply(os.path.exists)
if not data['exists'].all():
    print("Il y a des fichiers manquants après ajustement.")
    missing_files = data[~data['exists']]
    print(f"Nombre de fichiers manquants : {missing_files.shape[0]}")
    print(missing_files['img_paths'].head())
else:
    print("Tous les fichiers existent après ajustement.")
    # Procede uniquement si tous les fichiers existent
    data = data[data['exists']]  # Garde seulement les lignes avec des fichiers existants

# Ajouter une nouvelle colonne 'goodbook' basée sur la note des étoiles
print("------ Ajout de la colonne goodbook ------")
data['goodbook'] = (data['book_depository_stars'] >= 4.5).astype(int)
data['goodbook'] = data['goodbook'].astype(str)

# Séparer les données en ensemble d'entraînement et de test
print("------ Séparation des données train et test ------")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Préparer les générateurs de données
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='img_paths',
    y_col='goodbook',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='img_paths',
    y_col='goodbook',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Création du modèle MobileNetV2
print("------ Création du modèle ------")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Gel des couches convolutives pour le fine-tuning

inputs = Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compilation du modèle
print("------ Compilation du modèle ------")
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

# Entraînement du modèle
print("------ Entraînement du modèle ------")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_data) // 32,
    callbacks=[early_stopping]
)

# Calcul du F1-score à la fin de l'entraînement
print("------ Calcul du F1-score ------")
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype(int)
f1 = f1_score(y_true, y_pred)
print(f'Test F1-score: {f1}')

# Sauvegarde du modèle en format TFLite pour réduction de taille
print("------ Sauvegarde du modèle TFLite ------")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('C:/Users/abdou/Desktop/gp/data/modele_couverture_livre.tflite', 'wb') as f:
    f.write(tflite_model)
