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


# Non-Maximum Suppression (NMS) - Überlappende Rechtecke reduzieren
def non_maximum_suppression(boxes, confidences, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(confidences)

    selected_boxes = []
    while len(idxs) > 0:
        i = idxs[-1]
        selected_boxes.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap_area = w * h

        iou = overlap_area / (areas[i] + areas[idxs[:-1]] - overlap_area)

        idxs = idxs[np.where(iou <= overlap_thresh)[0]]

    return selected_boxes


def predict_sign(roi, model, debug_mode=False):
    # ROI-Bild in ein PIL-Image umwandeln
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Bild in einen Speicherpuffer schreiben (statt auf die Festplatte zu speichern)
    img_buffer = io.BytesIO()
    roi_pil.save(img_buffer, format="PNG")  # PNG, um keine Verluste zu haben
    img_buffer.seek(0)  # Speicherpuffer an den Anfang setzen

    # Bild mit keras `load_img` laden, genau wie in use_model.py
    img = tf.keras.preprocessing.image.load_img(img_buffer, target_size=(224, 224))

    # In ein numpy-Array umwandeln
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Umwandlung in Batch-Dimension (Modell erwartet einen Batch)
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage des Modells
    preds = model.predict(img_array)

    # Index der Klasse mit der höchsten Wahrscheinlichkeit
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]

    if debug_mode:
        print(f"Predicted Class: {class_labels[class_idx]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Shape of input image: {roi.shape}")
        print(f"ROI (cropped image) saved to 'live_debug/' with timestamp.")

        # Bild speichern mit einem Zeitstempel
        if not os.path.exists("live_debug"):
            os.makedirs("live_debug")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"live_debug/{timestamp}_{class_labels[class_idx]}.jpg"
        cv2.imwrite(file_name, roi)  # Bild speichern

    return interpret_sign(class_idx), confidence


def find_traffic_signs(frame, model, debug_mode=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    confidences = []

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if aspect_ratio > 0.5 and aspect_ratio < 2.0:
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )
            num_vertices = len(approx)

            if num_vertices == 4 or num_vertices > 4:
                roi = frame[y : y + h, x : x + w]

                predicted_class, confidence = predict_sign(roi, model, debug_mode)

                boxes.append((x, y, w, h))
                confidences.append(confidence)

    final_boxes = non_maximum_suppression(boxes, confidences, overlap_thresh=0.3)

    return final_boxes


cap = cv2.VideoCapture(0)
cv2.namedWindow("Traffic Sign Detection", cv2.WINDOW_NORMAL)

# ------------------------------------#
# Debug mode aktivieren/deaktivieren  #
# ------------------------------------#
debug_mode = True

# ------------------------------------#
# Video Capture Application           #
# ------------------------------------#
while True:
    ret, frame = cap.read()
    if not ret:
        break

    traffic_signs = find_traffic_signs(frame, model, debug_mode)

    for x, y, w, h in traffic_signs:
        roi = frame[y : y + h, x : x + w]

        predicted_class, confidence = predict_sign(roi, model, debug_mode)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{predicted_class}: {confidence*100:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Fenstergröße abrufen
    height, width = frame.shape[:2]
    win_width = cv2.getWindowImageRect("Traffic Sign Detection")[2]
    win_height = cv2.getWindowImageRect("Traffic Sign Detection")[3]

    # Frame auf Fenstergröße skalieren
    resized_frame = cv2.resize(frame, (win_width, win_height))

    cv2.imshow("Traffic Sign Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
