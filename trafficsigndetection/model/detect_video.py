import cv2
import numpy as np
import tensorflow as tf
import os
import time
from PIL import Image
import io

# Modell laden
try:
    model = tf.keras.models.load_model("traffic_sign_default_settings_model_final.h5")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Labels für das Modell
class_labels = {
    0: "Geschwindigkeitsbegrenzung (20km/h)",
    1: "Geschwindigkeitsbegrenzung (30km/h)",
    2: "Geschwindigkeitsbegrenzung (50km/h)",
    3: "Geschwindigkeitsbegrenzung (60km/h)",
    4: "Geschwindigkeitsbegrenzung (70km/h)",
    5: "Geschwindigkeitsbegrenzung (80km/h)",
    6: "Ende der Geschwindigkeitsbegrenzung (80km/h)",
    7: "Geschwindigkeitsbegrenzung (100km/h)",
    8: "Geschwindigkeitsbegrenzung (120km/h)",
    9: "Überholverbot",
    10: "Überholverbot für Fahrzeuge über 3,5t",
    11: "Vorfahrt an der nächsten Kreuzung",
    12: "Vorfahrtstraße",
    13: "Vorfahrt gewähren",
    14: "Stop",
    15: "Verbot für Fahrzeuge aller Art",
    16: "Verbot für Fahrzeuge über 3,5t",
    17: "Einfahrt verboten",
    18: "Allgemeine Gefahrstelle",
    19: "Kurve nach links",
    20: "Kurve nach rechts",
    21: "Doppelkurve",
    22: "Unebene Fahrbahn",
    23: "Schleudergefahr bei Nässe oder Schmutz",
    24: "Verengte Fahrbahn",
    25: "Baustelle",
    26: "Ampel",
    27: "Fußgänger",
    28: "Kinder",
    29: "Radfahrer",
    30: "Achtung Schnee oder Eisglätte",
    31: "Wildwechsel",
    32: "Ende aller Geschwindigkeits- und Überholverbote",
    33: "Rechts abbiegen",
    34: "Links abbiegen",
    35: "Geradeaus fahren",
    36: "Geradeaus oder rechts abbiegen",
    37: "Geradeaus oder links abbiegen",
    38: "Rechts vorbei",
    39: "Links vorbei",
    40: "Kreisverkehr",
    41: "Ende des Überholverbots",
    42: "Ende des Überholverbots für Fahrzeuge über 3,5t",
}

def interpret_sign(class_index):
    return class_labels.get(class_index, "Unbekanntes Schild")

def find_traffic_signs(frame, model, debug_mode=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio > 0.5 and aspect_ratio < 2.0:
            roi = frame[y:y+h, x:x+w]
            predicted_class, confidence = predict_sign(roi, model)
            if predicted_class != "Unbekanntes Schild":
                boxes.append((x, y, w, h, predicted_class, confidence))
    return boxes

def predict_sign(roi, model):
    roi_resized = cv2.resize(roi, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(roi_resized)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]
    return interpret_sign(class_idx), confidence

video_folder = "test_videos"
output_folder = "processed_videos"
os.makedirs(output_folder, exist_ok=True)

for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(output_folder, f"processed_{video_file}"), fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            traffic_signs = find_traffic_signs(frame, model)
            for x, y, w, h, predicted_class, confidence in traffic_signs:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_class}: {confidence*100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
        cap.release()
        out.release()

cv2.destroyAllWindows()
