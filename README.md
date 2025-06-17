# 🚗 TrafficSignDetection

Ein KI-gestütztes System zur Echtzeit-Erkennung von Verkehrszeichen aus Kamerabildern.  
Entwickelt im Rahmen der Diplomarbeit von Markus Brandstetter an der HTBL Hollabrunn, Schuljahr 2024/25.

---

## 📌 Projektbeschreibung

TrafficSignDetection nutzt ein selbst entwickeltes Convolutional Neural Network (CNN), um Verkehrszeichen automatisch zu erkennen und zu klassifizieren.  
Die Erkennung erfolgt in Echtzeit mithilfe von Live-Videodaten und moderner Bildverarbeitung.

---

## 🧠 Technologien

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- Pandas, NumPy, Matplotlib
- React Native (erste App-Prototypen)

---

## 📦 requirements.txt

```txt
tensorflow>=2.10
keras
opencv-python
numpy
pandas
matplotlib
Pillow
scikit-learn
```

---

## ▶️ Anwendung starten

```bash
python trafficsigndetection/model/detect.py
```

- Öffnet ein Kamerafenster  
- Erkennt Verkehrszeichen in Echtzeit  
- Zeichnet Begrenzungen/Rechtecke & zeigt Klasse + Konfidenz  

---

## 📁 Projektstruktur

```
TrafficSignDetection/
│ model/
│   ├── traffic_sign_default_settings_model.h5   # HDF5 des trainierten CNN
│   ├── train_default_settings.py                # Training & Evaluation           
│   ├── detect.py                                # Live-Erkennung using ROI, NMS and Preprocessing via Webcam
│   ├── Images/                                  # GTSRB-Datensatz
│   └── results/                                 # Training Accuracy and Results
```

---

## 🧪 Tests & Ergebnisse

- 100 Testbilder → 95 % korrekt erkannt  
- Live-Fahrten → Verkehrszeichen erfolgreich erkannt  
- Konfusionsmatrix zur Analyse der Klassengenauigkeit

---

![verkehr2](https://github.com/user-attachments/assets/e601f2a3-4c53-482e-8a48-4df726fea3d8)

---

## 🚀 Ausblick

- Mobile App-Integration (iOS/Android)  
- Deployment auf Edge-Devices (Raspberry Pi, Jetson)  
- Erweiterung um mehr Verkehrszeichenklassen

---

## 👤 Autor

**Markus Brandstetter**  
5BHITS, HTBL Hollabrunn  
TrafficSignDetection – Verkehrszeichenerkennungs-App
