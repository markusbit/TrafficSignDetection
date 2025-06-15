import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random

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
# 2. Modell laden
# =====================
model = load_model("traffic_sign_default_settings_model_final.h5")

# =====================
# Modell anwenden
# =====================
def predict_image(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    return class_idx, preds[0][class_idx]

# =====================
# 3. Test ohne Labels
# =====================
def predict_snippets(model, snippets_path, num_images=100):
    image_files = [f for f in os.listdir(snippets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.ravel()

    for i, filename in enumerate(selected_images):
        image_path = os.path.join(snippets_path, filename)
        predicted_label, confidence = predict_image(image_path, model)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

        axes[i].imshow(img)
        axes[i].set_title(f"{interpret_sign(predicted_label)}\n{confidence:.2f}", color='blue')
        axes[i].axis('off')

        print(f"File: {filename}")
        print(f"Predicted: {interpret_sign(predicted_label)} - Confidence: {confidence:.2f}")
        print("-" * 50)

    plt.tight_layout()
    plt.show()

# Beispielaufruf mit relativem Pfad
predict_snippets(model, 'test_videos/snippets')
