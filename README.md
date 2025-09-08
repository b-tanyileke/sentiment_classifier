# IMDb Sentiment Classifier

This project fine-tunes **DistilBERT** on the [IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to classify reviews as **Positive** or **Negative**.  
The final model is deployed as an interactive **Gradio app** on Hugging Face Spaces.  

## 📊 Project Overview
- Task: Binary sentiment classification (positive vs. negative reviews)
- Dataset: IMDb (50,000 movie reviews)
- Model: [DistilBERT](https://huggingface.co/distilbert-base-uncased) fine-tuned with Hugging Face `Trainer`
- Accuracy: ~91% on test set
- Deployment: [Live Gradio Demo on Hugging Face Spaces](https://huggingface.co/spaces/NkTanyileke/imdb-sentiment-app)


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
pip install -r requirements.txt
```

## 🚀 Usage

1. You can fine-tune DistilBERT yourself using the training notebook in text_classification.ipynb

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


## 🔗 Model & Demo Links

📦 Fine-tuned Model: [Hugging Face Hub](https://huggingface.co/NkTanyileke/imdb-sentiment-model)

🎨 Live Demo: [Gradio App](https://huggingface.co/spaces/NkTanyileke/imdb-sentiment-app)


## 📂 Repository Structure

imdb-sentiment-classifier/
    text_classification.ipynb       training notebook
    app.py                          Gradio demo
    requirements.txt                dependencies
    README.md                       this file
    .gitignore                      ignores cache/checkpoints
