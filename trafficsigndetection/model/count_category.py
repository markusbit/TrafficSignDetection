import os
import matplotlib.pyplot as plt
from collections import Counter

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

def count_files_in_subfolders(directory):
    category_counts = {label: 0 for label in class_labels.values()}
    
    for category in os.listdir(directory):
        if category.isdigit() and int(category) in class_labels:
            category_name = class_labels[int(category)]
            category_path = os.path.join(directory, category)
            if os.path.isdir(category_path):
                num_files = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
                category_counts[category_name] = num_files
    
    return category_counts

def plot_category_counts(category_counts, sorted_order=True):
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    if sorted_order:
        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i])
        categories = [categories[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 5))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Kategorien')
    plt.ylabel('Anzahl der Dateien')
    plt.xticks(rotation=45, ha='right')
    plt.title('Anzahl der Dateien pro Kategorie')
    plt.show()

if __name__ == "__main__":
    directory = "./Images"  # Ordner anpassen
    category_counts = count_files_in_subfolders(directory)
    
    # 1. Sortierte Darstellung
    plot_category_counts(category_counts, sorted_order=True)
    
    # 2. Unsortierte Darstellung
    plot_category_counts(category_counts, sorted_order=False)
