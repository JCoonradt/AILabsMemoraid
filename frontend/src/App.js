import { useState, useRef } from "react";
import './App.css';

function App() {
  const [transcription, setTranscription] = useState("");
  const [tasks, setTasks] = useState([]);
  const [isRecording, setIsRecording] = useState(false);

  const recognitionRef = useRef(null);
  const taskCounterRef = useRef(1);

  // Speech recognition setup
  const initializeSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Sorry, your browser doesn't support speech recognition.");
      return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          setTranscription((prev) => prev + transcript + " ");
          // extractTasks(transcript);
        }
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    return recognition;
  };

  const handleStart = () => {
    if (!recognitionRef.current) {
      recognitionRef.current = initializeSpeechRecognition();
    }
    recognitionRef.current.start();
    setIsRecording(true);
  };

  const handleStop = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsRecording(false);
  };



  const toggleTaskCompletion = (id) => {
    setTasks((prevTasks) =>
      prevTasks.map((task) =>
        task.id === id ? { ...task, completed: !task.completed } : task
      )
    );
  };
  
  
  
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const processText = async() => {
    setLoading(true);
    setError(null);

    try{
      const response = await fetch("http://127.0.0.1:5000/process", {
        method: "POST",
        headers: {
          "Content-Type" : "application/json",
        },
        body: JSON.stringify({text: transcription})
      })

      const data = await response.json();
      if (!response.ok){
        throw new Error(data.error || "Failed to process text");
      }

      setResult(data);
    }catch(err){
      setError(err.message)
    } finally{
      setLoading(false);
    }
  } 

  return (
    





    
   <div className="container">
        <div className="app">
      <h1>Memo-ry: Empowering Independence for People with Dementia</h1>
      <p className="description">
        Memo-ry is an innovative app designed to support individuals with
        dementia by recording and processing audio from daily conversations. It
        listens for tasks, demands, or instructions given to the person, then
        automatically generates an organized to-do list based on the recognized
        tasks.
      </p>

      <div className="button-container">
        <button onClick={handleStart} disabled={isRecording}>
          Start Recording
        </button>
        <button onClick={handleStop} disabled={!isRecording}>
          Stop Recording
        </button>
      </div>

      <div>
        <h2>Transcription:</h2>
        <p className="transcription">{transcription}</p>
      </div>

      <div>
        <h2>Extracted Tasks:</h2>
        <ul className="tasks">
          {tasks.map((task) => (
            <li
              key={task.id}
              style={{
                textDecoration: task.completed ? "line-through" : "none",
              }}
            >
              <input
                type="checkbox"
                checked={task.completed}
                onChange={() => toggleTaskCompletion(task.id)}
              />
              {task.text}
            </li>
          ))}
        </ul>
      </div>
    </div>

    <iframe src="https://calendar.google.com/calendar/embed?height=600&wkst=1&ctz=America%2FNew_York&showPrint=0&mode=AGENDA&src=NDNiOTUxMmRiZWY0ODc1ZDBhYTU3NGI4NjgxZjQxZDdmMjM3ZDZlYjJhYzJkN2JhZTRhODRjN2M1MTc5MzJiM0Bncm91cC5jYWxlbmRhci5nb29nbGUuY29t&color=%23E67C73"  width="800" height="600" frameborder="0" scrolling="no"></iframe>

   <div className="card">
     <h1 className="card-title">BERT Text Processing Workshop</h1>
     <div className="input-area">
       <textarea
         className="textarea"
         value={inputText}
         onChange={(e) => setInputText(e.target.value)}
         placeholder="Enter text to process..."
       />
     </div>
     <button
       className="button"
       onClick={processText}
       disabled={loading || !inputText.trim()}
     >
       {loading ? "Processing..." : "Process Text"}
     </button>
     {error && <div className="error">Error: {error}</div>}
     {result && (
       <div className="results">
         <h3 className="results-title">Results:</h3>
         <div className="results-content">
           {JSON.stringify(result, null, 2)}
         </div>
       </div>
     )}
   </div>
 </div>






  );
}

export default App;
