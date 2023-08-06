import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import io
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Définir le nom de la police
FONT_NAME = "Optima"

breed_names = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier',
               'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog',
               'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres',
               'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua',
               'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher',
               'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer', 'Gordon_setter',
               'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter',
               'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel',
               'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog',
               'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog',
               'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki',
               'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih-Tzu',
               'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier',
               'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier',
               'affenpinscher', 'basenji', 'basset', 'beagle', 'black-and-tan_coonhound', 'bloodhound', 'bluetick',
               'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie',
               'curly-coated_retriever', 'dhole', 'dingo', 'flat-coated_retriever', 'giant_schnauzer',
               'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher',
               'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke',
               'silky_terrier', 'soft-coated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer', 'toy_poodle',
               'toy_terrier', 'vizsla', 'whippet', 'wire-haired_fox_terrier', 'Mexican_hairless', 'Yorkshire_terrier',
               'golden_retriever']

model = tf.keras.models.load_model(
    filepath="model_transfer_inceptionv3_v1.2", custom_objects=None, compile=True, options=None)

img_height, img_width = 299, 299


# Télécharger l'image depuis l'URL et la prétraiter
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Charger une image depuis un fichier et la prétraiter
def preprocess_image_from_file(file_path):
    img = Image.open(file_path)
    img = img.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Afficher la prédiction à partir d'un URL
def show_prediction(image_url):
    processed_image = preprocess_image_from_url(image_url)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_breed = breed_names[predicted_class]

    # Calcul du pourcentage de confiance
    confidence = np.max(prediction) * 100

    img = Image.open(io.BytesIO(requests.get(image_url).content))
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)

    label_img.config(image=img)
    label_img.image = img
    label_pred.config(text="{} (Confiance: {:.2f}%)".format(predicted_breed, confidence))


# Ouvrir une boîte de dialogue pour choisir un fichier et afficher la prédiction
def choose_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        show_prediction_from_file(file_path)


# Afficher la prédiction à partir d'un fichier
def show_prediction_from_file(file_path):
    processed_image = preprocess_image_from_file(file_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_breed = breed_names[predicted_class]

    # Calcul du pourcentage de confiance
    confidence = np.max(prediction) * 100

    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)

    label_img.config(image=img)
    label_img.image = img
    label_pred.config(text="{} (Confiance: {:.2f}%)".format(predicted_breed, confidence))


# Interface graphique
root = tk.Tk()
root.title("Dog Breed Classifier")
root.geometry("900x500")
root.configure(bg="#222222")

# Centrer la fenêtre sur l'écran
window_width = 500
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Configurer la police pour l'ensemble des textes
font = (FONT_NAME, 16)  # Taille de la police : 12

frame = tk.Frame(root, bg="#222222")
frame.pack(pady=20)

label_url = tk.Label(frame, text="Entrez l'URL du chien:", bg="#222222", fg="white", font=font)
label_url.pack()

entry_url = tk.Entry(frame, bg="#444444", fg="white", font=font, borderwidth=0, highlightbackground="#444444",
                     highlightcolor="#666666", highlightthickness=2)
entry_url.pack(pady=5)

btn_url = tk.Button(frame, text="Afficher la prédiction (URL)", command=lambda: show_prediction(entry_url.get()),
                    bg="#444444", fg="white", font=font, relief=tk.FLAT, highlightbackground="#444444",
                    highlightcolor="#666666", highlightthickness=2)
btn_url.pack(pady=10)

btn_file = tk.Button(frame, text="Choisir un fichier", command=choose_file, bg="#444444", fg="white",
                     font=font, relief=tk.FLAT, highlightbackground="#444444", highlightcolor="#666666",
                     highlightthickness=2)
btn_file.pack(pady=10)

# Initialiser une image vide pour le label_img
empty_image = Image.new('RGB', (300, 300), color='#222222')
empty_image = ImageTk.PhotoImage(empty_image)

label_img = tk.Label(frame, image=empty_image, bg="#222222")
label_img.pack(pady=10)

label_pred = tk.Label(frame, text="", bg="#222222", fg="white", font=font, anchor='n')
label_pred.pack()

root.mainloop()
