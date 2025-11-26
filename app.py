import threading
import queue
import json
import csv
import os
import time
from datetime import datetime

from flask import Flask, Response, render_template_string
from pythonosc import dispatcher, osc_server

# --- Configuration ---
CSV_PATH = "muse_stream.csv"
OSC_IP = "0.0.0.0"
OSC_PORT = 3001

# --- Global Shared Queue ---
# This connects the OSC thread to the Flask thread
data_queue = queue.Queue()

# --- Flask App Setup ---
app = Flask(__name__)

# --- HTML Template (Embedded for single-file convenience) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Muse OSC Stream</title>
    <style>
        body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f4f4f9; }
        h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 0.5rem; }
        .status-bar { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; background-color: #ccc; border-radius: 50%; margin-right: 8px; }
        .status-indicator.active { background-color: #28a745; box-shadow: 0 0 8px #28a745; }
        
        #log-container {
            background: #1e1e1e; color: #00ff00; font-family: monospace;
            height: 500px; overflow-y: auto; padding: 1rem;
            border-radius: 8px; border: 1px solid #333;
            font-size: 0.9rem;
        }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #333; padding-bottom: 2px; }
        .log-ts { color: #888; margin-right: 10px; }
        .log-addr { color: #4db8ff; font-weight: bold; margin-right: 10px; }
        .log-val { color: #fff; }
    </style>
</head>
<body>
    <h1>Muse Brainwave Monitor</h1>
    
    <div class="status-bar">
        <div>
            <span id="connection-status" class="status-indicator"></span>
            <span id="status-text">Waiting for stream...</span>
        </div>
        <div>
            <strong>CSV File:</strong>And <em>{{ csv_path }}</em>
        </div>
    </div>

    <div id="log-container">
        </div>

    <script>
        const logContainer = document.getElementById('log-container');
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('status-text');
        let maxLines = 200; // Keep DOM light

        // Connect to the Flask Stream using Server-Sent Events (SSE)
        const eventSource = new EventSource("/stream");

        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            
            // Update Status
            statusDot.classList.add('active');
            statusText.innerText = "Receiving Data Live";

            // Create Log Entry
            const div = document.createElement('div');
            div.className = 'log-entry';
            div.innerHTML = `
                <span class="log-ts">${data.timestamp.split('T')[1]}</span>
                <span class="log-addr">${data.address}</span>
                <span class="log-val">${JSON.stringify(data.values)}</span>
            `;

            // Append and auto-scroll
            logContainer.appendChild(div);
            logContainer.scrollTop = logContainer.scrollHeight;

            // Prune old entries to prevent browser lag
            if (logContainer.children.length > maxLines) {
                logContainer.removeChild(logContainer.firstChild);
            }
        };

        eventSource.onerror = function() {
            statusDot.classList.remove('active');
            statusText.innerText = "Connection Lost / Reconnecting...";
        };
    </script>
</body>
</html>
"""

# --- Backend Logic ---

def write_header():
    """Create CSV with header if not exists."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "address", "values"])

def log_to_csv(timestamp, address, args):
    """Append one row instantly."""
    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, address, list(args)])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def osc_handler(address: str, *args):
    """
    Triggered by pythonosc when data arrives.
    1. Logs to CSV.
    2. Puts data into Queue for Flask.
    """
    timestamp = datetime.now().isoformat()
    
    # Prepare data object
    data_packet = {
        "timestamp": timestamp,
        "address": address,
        "values": args
    }

    # 1. Log to CSV
    log_to_csv(timestamp, address, args)
    
    # 2. Push to Queue (Non-blocking put)
    data_queue.put(data_packet)
    
    # Optional: Print to server console as well
    # print(f"OSC: {address} {args}", flush=True)

def start_osc_server():
    """Runs the blocking OSC server."""
    disp = dispatcher.Dispatcher()
    disp.map("*", osc_handler)
    
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
    print(f"--- STARTED OSC SERVER on {OSC_IP}:{OSC_PORT} ---", flush=True)
    server.serve_forever()

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the dashboard."""
    return render_template_string(HTML_TEMPLATE, csv_path=CSV_PATH)

@app.route('/stream')
def stream():
    """
    Generator function for Server-Sent Events (SSE).
    Yields data from the queue to the browser.
    """
    def event_stream():
        while True:
            # Block until data is available in the queue
            data_packet = data_queue.get() 
            
            # Format as SSE (data: <payload>\n\n)
            json_data = json.dumps(data_packet)
            yield f"data: {json_data}\n\n"
            
    return Response(event_stream(), mimetype="text/event-stream")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup CSV
    write_header()

    # 2. Start OSC Server in a Background Thread
    # We use a daemon thread so it automatically dies when the main Flask app stops
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    # 3. Start Flask Server (Blocking) on Main Thread
    print(f"--- STARTED WEB DASHBOARD on http://localhost:5000 ---", flush=True)
    
    # debug=True is not compatible with threaded OSC in the same file comfortably
    # so we turn it off for production-like stability here.
    app.run(host="0.0.0.0", port=3000, debug=False)