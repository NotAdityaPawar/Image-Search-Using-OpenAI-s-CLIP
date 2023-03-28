import os
import PySimpleGUI as sg
import torch
from PIL import Image, ImageTk
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import clip


def preprocess(image):
    transform = Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)


def preprocess_images(image_paths, model, device):
    image_features = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_tensor)
        features = features.flatten()
        image_features.append(features.cpu().numpy())
    return np.stack(image_features)


def search(query, image_paths, model, device):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = preprocess_images(image_paths, model, device)
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    results = [(image_paths[idx], values[i].item()) for i, idx in enumerate(indices)]
    return results


model, preprocess = clip.load('ViT-B/32', device='cpu')


# splash_layout = [
#     [sg.Text("Loading...", font=("Helvetica", 20))]
# ]

# # Create a splash screen window
# splash_window = sg.Window("Image search", splash_layout, no_titlebar=True, grab_anywhere=True)

# # Simulate loading time
# for i in range(3):
#     sg.PopupQuick(f"Loading... {'.' * i}")
#     sg.PopupQuick(f"Loading... {'.' * (i + 1)}")

# # Close the splash screen window
# splash_window.close()



sg.theme('LightGrey')
layout = [
    [sg.Text("Search query:"), sg.Input(key="-QUERY-"), sg.Button("Search")],
    [sg.Text("Search results:"), sg.Listbox(values=[], size=(50, 10), key="-RESULTS-")],
    [sg.Image(key="-IMAGE-", size=(224, 224))]
]

window = sg.Window("Image search", layout)

def search_button_callback(query, image_paths, model, device, window):
    results = search(query, image_paths, model, device)
    window["-RESULTS-"].update(values=[f"{result[0]} ({result[1]})" for result in results])
    print(results)
    if results:
        image = Image.open(results[0][0]).resize((224, 224))
        window["-IMAGE-"].update(data=ImageTk.PhotoImage(image))

        def result_listbox_callback(event):
            if len(event) > 0:
                index = event[0]
                image = Image.open(results[index][0]).resize((224, 224))
                window["-IMAGE-"].update(data=ImageTk.PhotoImage(image))

        window["-RESULTS-"].bind('<Double-Button-1>', result_listbox_callback)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "Search":
        query = values["-QUERY-"]
        image_paths = [os.path.join("dataset/", filename) for filename in os.listdir("dataset/") if filename.endswith(".jpg")]
        search_button_callback(query, image_paths, model, "cpu", window)

window.close()
