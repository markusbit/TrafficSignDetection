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
python scripts/detect_signs.py
```

- Öffnet ein Kamerafenster  
- Erkennt Verkehrszeichen in Echtzeit  
- Zeichnet Begrenzungen/Rechtecke & zeigt Klasse + Konfidenz  

---

## 📁 Projektstruktur

```
TrafficSignDetection/
│
├── model/
│   └── traffic_sign_model.h5      # HDF5 des trainierten CNN
├── data/
│   ├── Images/                    # GTSRB-Datensatz
│   └── test_videos/               # Aufnahmen für Live-Tests
├── scripts/
│   ├── train_model.py             # Training & Evaluation
│   ├── detect_signs.py            # Live-Erkennung via Webcam/DroidCam
│   └── utils.py                   # ROI, NMS, Preprocessing
├── results/
│   └── confusion_matrix_results.xlsx
├── README.md
└── requirements.txt
```

---

## 🧪 Tests & Ergebnisse

- 100 Testbilder → 95 % korrekt erkannt  
- Live-Fahrten → Verkehrszeichen erfolgreich erkannt  
- Konfusionsmatrix zur Analyse der Klassengenauigkeit

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
