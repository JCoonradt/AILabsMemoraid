from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

app = Flask(__name__)
CORS(app)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Perform inference
    with torch.no_grad():
        logits_repeat, logits_relative_time, logits_exact_time = model(input_ids, attention_mask)
        
        # Get predicted classes
        predicted_repeat = torch.argmax(logits_repeat, dim=-1).item()
        predicted_relative_time = torch.argmax(logits_relative_time, dim=-1).item()
        predicted_exact_time = torch.argmax(logits_exact_time, dim=-1).item()
    
    return predicted_repeat, predicted_relative_time, predicted_exact_time




class CustomBertModel(nn.Module):
    def __init__(self):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.fc_repeat = nn.Linear(hidden_size, 9)  # Match saved model (e.g., 7 days + daily + no)
        self.fc_relative_time = nn.Linear(hidden_size, 3)  # Morning, Afternoon, Night
        self.fc_exact_time = nn.Linear(hidden_size, 2)  # True/False

        # Initialize position_ids to avoid missing key errors
        self.bert.embeddings.position_ids = torch.arange(512).expand((1, -1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits_repeat = self.fc_repeat(pooled_output)
        logits_relative_time = self.fc_relative_time(pooled_output)
        logits_exact_time = self.fc_exact_time(pooled_output)
        return logits_repeat, logits_relative_time, logits_exact_time
    
model = CustomBertModel()
model.load_state_dict(torch.load("AIM_Lab_11-16.pth", weights_only=True,  map_location=torch.device('cpu')))
model.eval()

#this is supposed to simulate a real bert output
def get_dummy_embeddings(text):
    np.random.seed(len(text))
    return np.random.randn(768).tolist()

#load up BERT model (if this wasn't using dummy data)

@app.route("/process", methods=["POST"])
def process_text():
    try:
        data = request.json
        text = data.get("text","")
        # Step 4: Test the model with an example

        predicted_repeat, predicted_relative_time, predicted_exact_time = predict(text)

        # Map predictions to human-readable labels
        repeat_mapping = {3: "No", 2: "Monday", 7: "Tuesday", 8: "Wednesday", 6: "Thursday", 1: "Friday", 4: "Saturday", 5: "Sunday", 0: "Daily"}
        relative_time_mapping = {1: "Morning", 0: "Afternoon", 2: "Night"}
        exact_time_mapping = {0: "False", 1: "True"}
        
        return jsonify({
            "success": True,
            "embeddings": [repeat_mapping[predicted_repeat], relative_time_mapping[predicted_relative_time], exact_time_mapping[predicted_exact_time]],
            "message": "Text processed succesfully",
            "text_length": len(text)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug = True, port=5000)