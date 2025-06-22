
import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Define class names
class_names = ["cup_cakes", "french_fries", "omelette", "pizza", "samosa", "steak", "sushi"]

# Load model and transforms
model, model_transforms = create_model(num_classes=len(class_names))
model.load_state_dict(
    torch.load(
        f="my_model.pth",
        map_location=torch.device("cpu")
    )
)

def predict(img) -> Tuple[Dict, float]:
    '''Transforms and performs a prediction on img and returns prediction and time taken'''
    # Start timer
    start_time = timer()

    # Transform the image and add batch dimension
    img = model_transforms(img).unsqueeze(0)

    # Evaluate mode and inference context
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)

    # Prepare predictions
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Time taken
    pred_time = round(timer() - start_time, 3)

    return pred_labels_and_probs, pred_time

# Gradio UI setup
title = "FoodVision App🍴"
description = "An Efficient Computer Vision Model to classify images for 7 types of food"
article = "Created with Python"

# Load examples
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=len(class_names), label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

demo.launch()
