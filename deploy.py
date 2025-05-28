from flask import Flask, render_template, request  # Import necessary libraries for web framework and request handling
import pickle  # For loading the label encoder
import torch  # For model inference using PyTorch

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and tokenizer from the Hugging Face model hub
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained sequence classification model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("my_model")
label_encoder = pickle.load(open('label_encoder.sav', 'rb'))
tokenizer = AutoTokenizer.from_pretrained("my_model")

@app.route('/')
def home():
    """
    Home route to display the main page.

    This function renders the index.html template and initializes a blank result variable
    to pass to the HTML page. The page can be used for displaying prediction results.
    """
    result = ''  # Initialize an empty result variable
    return render_template('index.html', **locals())  # Render the main page with the result

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """
    Prediction route to handle text input and generate model predictions.

    This function processes the user input text, tokenizes it, and makes predictions using
    the loaded model. The result is then decoded using a label encoder and returned to the
    user via the rendered page.
    """
    # Get the text input from the form on the web page
    text_class = [request.form['text_class']]  # Retrieve the user input text
    
    # Tokenize the input text and prepare it for model inference
    new_encodings = tokenizer(text_class, truncation=True, padding=True, return_tensors="pt")
    
    # Get the device (CPU, CUDA, or MPS) for model inference
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the selected device
    
    # Move the tokenized input to the same device as the model
    if 'token_type_ids' in new_encodings:
        del new_encodings['token_type_ids']
    new_encodings = {key: val.to(device) for key, val in new_encodings.items()}

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**new_encodings)  # Run the model on the input
        predictions = torch.argmax(outputs.logits, dim=1)  # Get the predicted class (max logit value)

    # Decode the predictions into human-readable labels
    decoded_preds = label_encoder.inverse_transform(predictions.cpu().numpy())
    result = decoded_preds[0]

    return render_template('index.html', **locals())  # Render the page with the prediction result

# Run the Flask app when the script is executed
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Start the Flask web server