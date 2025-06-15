import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model  # Import plot_model
import visualkeras

# Dummy for data augmentation (optional)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# Klassenanzahl (z. B. Verkehrszeichen)
class_labels = list(range(43))  # z. B. 43 Klassen

# === Modell definieren ===
model = Sequential([
    data_augmentation,
    Rescaling(1.0 / 255),
    
    Conv2D(128, (3, 3), activation="relu", input_shape=(56, 56, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dense(len(class_labels), activation="softmax"),
])

model.build(input_shape=(None, 56, 56, 3))
model.summary()

# Visualize model with plot_model
plot_model(model, to_file="model_visualization.png", show_shapes=True, show_layer_names=True)

# visualkeras.layered_view(model)

print("Model visualization saved as 'model_visualization.png'.")
