"""
IMDb Sentiment Classifier Web App

This script loads a fine-tuned DistilBERT model trained on the IMDb movie review dataset
and serves it as an interactive Gradio web app. Users can input a movie review and receive
a prediction indicating whether the sentiment is Positive or Negative, along with the
confidence score.
"""

# import necessary libraries
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model directory
model_dir = "./imdb_model"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# move model to gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# set model in evaluation mode
model.eval()

# define prediciton function
def predict_sentiment(text):
    """ Function gets text as input processes it,
        Makes predicitons using the model and return class probabilities
    """
    # tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # move to gpu
    inputs = {k:v.to(device) for k, v in inputs.items()}

    # run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    return {"Positive": probs[0][1].item(), "Negative": probs[0][0].item()}

# 3️⃣ Create Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,       # prediction function
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review here..."),
    outputs=gr.Label(num_top_classes=2),
    title="IMDb Sentiment Classifier",
    description="The model will classify a movie preview as Positive or Negative.",
    examples=[
        ["The movie was fantastic! I loved it."],
        ["Terrible movie, waste of time."],
    ]
)

# 4️⃣ Launch the app
interface.launch()
