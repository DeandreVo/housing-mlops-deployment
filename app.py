import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("model.pkl")

def predict_price(area):
    input_data = np.array([[float(area)]])
    prediction = model.predict(input_data)
    return f"Predicted Price: ${prediction[0]:,.2f}"

# Define Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=gr.Number(label="Area (sq ft)"),
    outputs=gr.Textbox(label="Predicted Price"),
    title="Housing Price Predictor"
)

interface.launch()
