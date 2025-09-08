# IMDb Sentiment Classifier

This project fine-tunes **DistilBERT** on the **IMDb movie review dataset** to classify reviews as **Positive** or **Negative**. The model is wrapped in a **Gradio web app** for interactive predictions.

## 📦 Features
- Fine-tuned DistilBERT for sentiment analysis
- Interactive Gradio demo for testing movie reviews
- Shows predicted sentiment with confidence score
- Easy deployment to Hugging Face Spaces

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/b-tanyileke/sentiment_classifier.git
cd sentiment_classifier
```

2. Create a Python environment (optional but recommended):
```bash
conda create -n torch-env python=3.10
conda activate torch-env
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate scikit-learn gradio
```

## 🚀 Usage

1. Train the model by running notebook cells:

2. Launch the Gradio app:
```bash
python app.py
```
3. Enter any movie review in the text box to get prediction.

## 📊 Example Predictions

| Review                           | Prediction       |
| -------------------------------- | ---------------- |
| "The movie was fantastic!"       | Positive (98.5%) |
| "Terrible movie, waste of time." | Negative (95.3%) |

## 🔗 Deployment

- Push to Hugging Face Spaces:

    1. Login: huggingface-cli login

    2. Push repo with app.py and imdb_model folder

    3. The Gradio app will be live online
