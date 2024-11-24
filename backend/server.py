from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from datetime import datetime, timedelta
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
        add_task_from_string(text)
        return jsonify({
            "success": True,
            "embeddings": [repeat_mapping[predicted_repeat], relative_time_mapping[predicted_relative_time], exact_time_mapping[predicted_exact_time]],
            "message": "Text processed succesfully",
            "text_length": len(text)
        })
    

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/calendar.events"]



def get_calendar_id(service, calendar_name="AI@MIT Test"):
    """Fetches the calendar ID for a given calendar name."""
    calendars_result = service.calendarList().list().execute()
    calendars = calendars_result.get("items", [])

    for calendar in calendars:
        if calendar["summary"] == calendar_name:
            return calendar["id"]
    return None

def create_event(service, calendar_id, task, start_time, end_time):
    """Creates an event in the calendar."""
    event = {
        "summary": task,
        "start": {
            "dateTime": start_time,
            "timeZone": "America/New_York"
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "America/New_York"
        },
    }
    return service.events().insert(calendarId=calendar_id, body=event).execute()

def add_task_to_calendar(task, start_time, end_time, calendar_name="AI@MIT Test"):
    """Adds a new task to the specified calendar."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("calendar", "v3", credentials=creds)

        # Get the calendar ID
        calendar_id = get_calendar_id(service, calendar_name)
        if not calendar_id:
            print(f"Calendar '{calendar_name}' not found.")
            return

        # Create the event
        event = create_event(service, calendar_id, task, start_time, end_time)
        print(f"Event created: {event.get('htmlLink')}")

    except HttpError as error:
        print(f"An error occurred: {error}")

def format_time_for_calendar(start_time_input, duration_in_minutes=30):
    """
    Converts a human-readable time like '3:00 PM' into ISO 8601 format for Google Calendar.
    :param start_time_input: The start time as a string (e.g., '3:00 PM').
    :param duration_in_minutes: Duration of the event in minutes.
    :return: A tuple (start_time, end_time) in ISO 8601 format.
    """
    try:
        # Parse the input time (assuming the date is today)
        now = datetime.now()
        start_time = datetime.strptime(start_time_input, "%I:%M %p").replace(
            year=now.year, month=now.month, day=now.day
        )

        # Calculate the end time based on the duration
        end_time = start_time + timedelta(minutes=duration_in_minutes)

        # Return the times in ISO 8601 format
        return start_time.isoformat(), end_time.isoformat()
    except ValueError as e:
        raise ValueError(f"Invalid time format: {start_time_input}. Use 'h:mm AM/PM'.") from e
    
def get_current_time_for_calendar():
    """
    Returns the current time in ISO 8601 format with timezone information for Google Calendar.
    Example output: '2024-11-23T15:30:00-05:00'
    """
    # Get the current local time
    now = datetime.now()

    # Format the current time to include timezone information
    current_time = now.isoformat()

    return current_time

def extract_time(input_string):
    """
    Extracts time from a string.
    Example: 'Walk the dog at 3:00 PM' -> '3:00 PM'
    """
    # Regular expression to extract time in the format of "h:mm AM/PM"
    time_match = re.search(r"\b\d{1,2}:\d{2} (AM|PM)\b", input_string, re.IGNORECASE)

    if not time_match:
        raise ValueError("No valid time found in the input string.")

    # Extract the time
    return time_match.group()

def add_task_from_string(input_string, duration_in_minutes=30, calendar_name="AI@MIT Test"):
    """
    Adds the full command as the task name and extracts the time to schedule the task on Google Calendar.
    :param input_string: Input string containing the task and time (e.g., 'Walk the dog at 3:00 PM').
    :param duration_in_minutes: Duration of the task in minutes.
    :param calendar_name: Name of the Google Calendar to add the event to.
    """
    try:
        # Extract the time from the string
        start_time_input = extract_time(input_string)

        # Format the time for Google Calendar
        start_time, end_time = format_time_for_calendar(start_time_input, duration_in_minutes)

        # Add the full input string as the task name
        add_task_to_calendar(input_string, start_time, end_time, calendar_name)

        print(f"Task '{input_string}' added to the calendar starting at {start_time_input}.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    app.run(debug = True, port=5000)