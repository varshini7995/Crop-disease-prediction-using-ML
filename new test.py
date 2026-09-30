import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ===============================
# LOAD MODEL
# ===============================
model = load_model("hybrid_cnn_vgg16_final.h5")

# ===============================
# CLASS NAMES (YOUR FINAL CLASSES)
# ===============================
CLASS_NAMES = [
    "Bacterial spot",
    "Black rot",
    "Common rust",
    "Powdery mildew",
    "Septoria leaf spot",
    "Normal"
]

# ===============================
# IMAGE PREDICTION FUNCTION
# ===============================
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)
    confidence = np.max(pred) * 100

    return CLASS_NAMES[class_id], confidence

# ===============================
# SELECT MULTIPLE IMAGES
# ===============================
def select_images():
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    for widget in frame.winfo_children():
        widget.destroy()

    for path in file_paths:
        disease, conf = predict_image(path)

        img = Image.open(path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(frame, image=img)
        panel.image = img
        panel.pack(side="left", padx=10, pady=10)

        label = tk.Label(
            frame,
            text=f"{disease}\nConfidence: {conf:.2f}%",
            font=("Arial", 10, "bold")
        )
        label.pack(side="left", padx=5)

# ===============================
# TKINTER GUI
# ===============================
root = tk.Tk()
root.title("Crop Disease Detection - Multi Image Testing")
root.geometry("1000x500")

btn = tk.Button(
    root,
    text="Select Multiple Images",
    command=select_images,
    font=("Arial", 14),
    bg="green",
    fg="white"
)
btn.pack(pady=20)

canvas = tk.Canvas(root)
canvas.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
scrollbar.pack(side="bottom", fill="x")

canvas.configure(xscrollcommand=scrollbar.set)

frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

root.mainloop()



