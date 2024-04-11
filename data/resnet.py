import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import SGD

# Charger les données
print("------ Chargement des données ------")
data = pd.read_csv('C:/Users/abdou/Desktop/gp/data/main_dataset.csv')
print(data.head())

# Ajouter une nouvelle colonne 'goodbook' basée sur la note des étoiles
print("------ Ajout de la colonne goodbook ------")
data['goodbook'] = (data['book_depository_stars'] >= 4.5).astype(int)
# Convertir la colonne 'goodbook' en chaînes de caractères
data['goodbook'] = data['goodbook'].astype(str)

# Séparer les données en ensemble d'entraînement et de test
print("------ Séparation des données train et test ------")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Générer des données d'entraînement pour les images
print("------ Génération des données d'entrainement pour les images ------")
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='img_paths',
    y_col='goodbook',
    target_size=(64, 64),  # Taille d'entrée de ResNet
    batch_size=32,
    class_mode='binary'
)

# Générer des données de test pour les images
print("------ Génération des données de test pour les images ------")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='img_paths',
    y_col='goodbook',
    target_size=(64, 64),  # Taille d'entrée de ResNet
    batch_size=32,
    class_mode='binary'
)

# Création du modèle ResNet50V2
print("------ Création du modèle ------")
input_shape = (64, 64, 3)  # Taille d'entrée de ResNet
image_input = Input(shape=input_shape, name='image_input')

# Utilisation de ResNet50V2 pré-entraîné
base_model = ResNet50V2(include_top=False, input_tensor=image_input, weights='imagenet') 
x = base_model.output

# Ajout de couches entièrement connectées pour la classification binaire
#x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.GlobalMaxPooling2D()(x)

x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=image_input, outputs=predictions)

# Compilation du modèle
print("------ Compilation du modèle ------")
model.compile(optimizer=LegacyAdam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
#model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    verbose=1, 
    restore_best_weights=True
)

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

#Sauvegarder le modèle
print("------ Sauvegarde du modèle ------")
model.save('C:/Users/abdou/Desktop/gp/data/modele_couverture_livre.h5')

