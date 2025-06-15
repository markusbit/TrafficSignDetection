import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil
import random
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd  # Importiere pandas für die Excel-Ausgabe

# =====================
# 1. Labels definieren
# =====================
class_labels = {
    0: 'Geschwindigkeitsbegrenzung (20km/h)',
    1: 'Geschwindigkeitsbegrenzung (30km/h)',
    2: 'Geschwindigkeitsbegrenzung (50km/h)',
    3: 'Geschwindigkeitsbegrenzung (60km/h)',
    4: 'Geschwindigkeitsbegrenzung (70km/h)',
    5: 'Geschwindigkeitsbegrenzung (80km/h)',
    6: 'Ende der Geschwindigkeitsbegrenzung (80km/h)',
    7: 'Geschwindigkeitsbegrenzung (100km/h)',
    8: 'Geschwindigkeitsbegrenzung (120km/h)',
    9: 'Überholverbot',
    10: 'Überholverbot für Fahrzeuge über 3,5t',
    11: 'Vorfahrt an der nächsten Kreuzung',
    12: 'Vorfahrtstraße',
    13: 'Vorfahrt gewähren',
    14: 'Stop',
    15: 'Verbot für Fahrzeuge aller Art',
    16: 'Verbot für Fahrzeuge über 3,5t',
    17: 'Einfahrt verboten',
    18: 'Allgemeine Gefahrstelle',
    19: 'Kurve nach links',
    20: 'Kurve nach rechts',
    21: 'Doppelkurve',
    22: 'Unebene Fahrbahn',
    23: 'Schleudergefahr bei Nässe oder Schmutz',
    24: 'Verengte Fahrbahn',
    25: 'Baustelle',
    26: 'Ampel',
    27: 'Fußgänger',
    28: 'Kinder',
    29: 'Radfahrer',
    30: 'Achtung Schnee oder Eisglätte',
    31: 'Wildwechsel',
    32: 'Ende aller Geschwindigkeits- und Überholverbote',
    33: 'Rechts abbiegen',
    34: 'Links abbiegen',
    35: 'Geradeaus fahren',
    36: 'Geradeaus oder rechts abbiegen',
    37: 'Geradeaus oder links abbiegen',
    38: 'Rechts vorbei',
    39: 'Links vorbei',
    40: 'Kreisverkehr',
    41: 'Ende des Überholverbots',
    42: 'Ende des Überholverbots für Fahrzeuge über 3,5t',
}

def interpret_sign(class_index):
    return class_labels.get(class_index, 'Unbekanntes Schild')

# =====================
# 2. Dataset Path
# =====================
dataset_path = r'Images'
batch_size = 32
img_size = (224, 224)

# =====================
# 3. Dataset laden
# =====================
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# =====================
# 4. Modell erstellen (mit EfficientNetB0)
# =====================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Wir frieren das Basis-Modell ein

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# 5. Modell trainieren
# =====================
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=[early_stopping]
)

# =====================
# 6. Modell speichern
# =====================
model.save("traffic_sign_pretrained_model.h5")
print("Modell gespeichert.")

# =====================
# 7. Konfusionsmatrix berechnen
# =====================
def generate_confusion_matrix(model, val_ds):
    # Wahrheitswerte und Vorhersagen sammeln
    true_labels = []
    predictions = []

    for images, labels in val_ds:
        # Vorhersagen des Modells
        preds = model.predict(images)
        predicted_class = np.argmax(preds, axis=1)
        
        # Labels der Validierungsbilder
        true_labels.extend(labels.numpy())
        predictions.extend(predicted_class)

    # Konfusionsmatrix berechnen
    cm = confusion_matrix(true_labels, predictions)

    # Genauigkeit pro Kategorie berechnen
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    # Speichern der Konfusionsmatrix und der Genauigkeit in eine Excel-Datei
    df_cm = pd.DataFrame(cm, index=list(class_labels.values()), columns=list(class_labels.values()))
    df_accuracy = pd.DataFrame({'Category': list(class_labels.values()), 'Accuracy': accuracy_per_class})

    # Excel-Datei speichern
    with pd.ExcelWriter('confusion_matrix_results.xlsx') as writer:
        df_cm.to_excel(writer, sheet_name='Confusion Matrix')
        df_accuracy.to_excel(writer, sheet_name='Accuracy per Category')

    # Konfusionsmatrix visualisieren
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()))
    plt.xlabel('Vorhergesagte Labels')
    plt.ylabel('Wahre Labels')
    plt.title('Konfusionsmatrix')
    plt.show()

# Konfusionsmatrix anzeigen und Excel-Datei speichern
generate_confusion_matrix(model, val_ds)
