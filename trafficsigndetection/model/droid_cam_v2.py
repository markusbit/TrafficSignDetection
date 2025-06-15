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

# Labels
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


def non_maximum_suppression(boxes, confidences, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(confidences)

    selected = []
    while len(idxs) > 0:
        last = idxs[-1]
        selected.append(boxes[last])
        suppress = [last]

        for i in idxs[:-1]:
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            overlap = (w * h) / areas[i]

            if overlap > overlap_thresh:
                suppress.append(i)

        idxs = np.delete(idxs, np.isin(idxs, suppress))

    return selected


def predict_sign(roi, model, debug_mode=False):
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_buffer = io.BytesIO()
    roi_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    img = tf.keras.preprocessing.image.load_img(img_buffer, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]

    return interpret_sign(class_idx), confidence


def find_traffic_signs(frame, model, debug_mode=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    confidences = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700:  # Kleinere Schilder zulassen
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 0.6 <= aspect_ratio <= 1.6:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) >= 4:
                roi = frame[y:y+h, x:x+w]
                predicted_class, confidence = predict_sign(roi, model, debug_mode)

                # Neue Schwelle: schon ab 0.6 anzeigen
                if confidence >= 0.45:
                    boxes.append((x, y, w, h))
                    confidences.append(confidence)

    return non_maximum_suppression(boxes, confidences, overlap_thresh=0.2)


# ----------------------------
# Kameraquelle
# DroidCam URL "http://192.168.0.100:4747/video"
# ----------------------------
CAMERA_URL = 1  # Für interne Kamera

cap = cv2.VideoCapture(CAMERA_URL)
cv2.namedWindow("Traffic Sign Detection", cv2.WINDOW_NORMAL)
debug_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kein Frame erhalten.")
        break

    traffic_signs = find_traffic_signs(frame, model, debug_mode)

    for (x, y, w, h) in traffic_signs:
        roi = frame[y:y+h, x:x+w]
        predicted_class, confidence = predict_sign(roi, model, debug_mode)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{predicted_class} ({confidence*100:.1f}%)",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    #win_w = cv2.getWindowImageRect("Traffic Sign Detection")[2]
    #win_h = cv2.getWindowImageRect("Traffic Sign Detection")[3]
    #frame_resized = cv2.resize(frame, (win_w, win_h))

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
