import os
import threading
import subprocess
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

# --- CONFIG -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "csv")
AUDIO_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "audio")

ALLOWED_CSV_EXTENSIONS = {"csv"}
ALLOWED_AUDIO_EXTENSIONS = {"mp3"}  # MP3 ONLY

os.makedirs(CSV_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_UPLOAD_FOLDER, exist_ok=True)

# Use your live Muse logger script here:
EEG_SCRIPT_NAME = "ece446.py"  # make sure this file is in the same folder as app.py

app = Flask(__name__)
app.secret_key = "change_this_to_something_random_and_secret"

# Socket.IO (threading mode is fine for dev)
socketio = SocketIO(app, async_mode="threading")

# --- STATE FOR EEG PROCESS ----------------------------------------------------

eeg_thread = None
eeg_process = None
eeg_running = False
eeg_lock = threading.Lock()


# --- HELPERS ------------------------------------------------------------------


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def timestamped_filename(original_name: str) -> str:
    """
    Add a timestamp so we don't overwrite earlier uploads:
    'eeg.csv' -> '20251124_142355_eeg.csv'
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = secure_filename(original_name)
    return f"{ts}_{safe}"


def eeg_reader():
    """
    Start ece446.py as a subprocess and stream its stdout
    to all connected clients via Socket.IO.
    """
    global eeg_process, eeg_running

    script_path = os.path.join(BASE_DIR, EEG_SCRIPT_NAME)

    try:
        if not os.path.exists(script_path):
            socketio.emit(
                "eeg_status",
                {
                    "running": False,
                    "message": (
                        f"EEG script not found at {script_path}. "
                        f"Make sure {EEG_SCRIPT_NAME} is in the same folder as app.py."
                    ),
                },
            )
            with eeg_lock:
                eeg_running = False
            return

        eeg_process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with eeg_lock:
            eeg_running = True

        socketio.emit(
            "eeg_status",
            {"running": True, "message": "EEG process started."},
        )

        assert eeg_process.stdout is not None
        for raw_line in eeg_process.stdout:
            line = raw_line.strip()
            if not line:
                continue

            # Just forward the raw line
            socketio.emit(
                "eeg_sample",
                {
                    "line": line,
                },
            )

    except Exception as e:
        socketio.emit(
            "eeg_status",
            {
                "running": False,
                "message": f"EEG reader error: {e}",
            },
        )
    finally:
        with eeg_lock:
            eeg_running = False
            eeg_process = None
        socketio.emit(
            "eeg_status",
            {"running": False, "message": "EEG process stopped."},
        )


# --- ROUTES -------------------------------------------------------------------


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page:
      - lets you (optionally) upload CSV and/or MP3
      - shows a separate Live Brainwaves section that works even if you don't upload anything
    """
    csv_filename = None
    audio_filename = None
    selected_mode = request.form.get("mode", "csv") if request.method == "POST" else None

    if request.method == "POST":
        mode = selected_mode or "csv"
        csv_file = request.files.get("csv_file")
        audio_file = request.files.get("audio_file")

        errors = []

        # For this version, uploads are optional. Only validate if user actually picks a file.
        if audio_file and audio_file.filename:
            if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                errors.append("Unsupported audio format. Please use an MP3 file (.mp3).")

        if mode == "csv":
            if csv_file and csv_file.filename:
                if not allowed_file(csv_file.filename, ALLOWED_CSV_EXTENSIONS):
                    errors.append("EEG file must have extension .csv.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("upload.html")

        # Save files if provided
        if csv_file and csv_file.filename:
            csv_filename = timestamped_filename(csv_file.filename)
            csv_path = os.path.join(CSV_UPLOAD_FOLDER, csv_filename)
            csv_file.save(csv_path)
            flash(f"CSV file uploaded as {csv_filename}", "success")

        if audio_file and audio_file.filename:
            audio_filename = timestamped_filename(audio_file.filename)
            audio_path = os.path.join(AUDIO_UPLOAD_FOLDER, audio_filename)
            audio_file.save(audio_path)
            flash(f"Audio file uploaded as {audio_filename}", "success")

        if not csv_filename and not audio_filename:
            flash("No files were uploaded. Brainwave logging still works without uploads.", "success")

        return render_template(
            "upload.html",
            csv_filename=csv_filename,
            audio_filename=audio_filename,
            selected_mode=mode,
        )

    # GET request
    return render_template("upload.html")


@app.route("/uploads/csv/<path:filename>")
def download_csv(filename):
    return send_from_directory(CSV_UPLOAD_FOLDER, filename, as_attachment=True)


@app.route("/uploads/audio/<path:filename>")
def get_audio(filename):
    return send_from_directory(AUDIO_UPLOAD_FOLDER, filename)


# --- SOCKET.IO EVENTS ---------------------------------------------------------


@socketio.on("start_live")
def handle_start_live():
    """Start ece446.py as a background process and stream its stdout."""
    global eeg_thread, eeg_running

    with eeg_lock:
        if eeg_running:
            emit(
                "eeg_status",
                {"running": True, "message": "EEG is already running."},
            )
            return

        eeg_thread = threading.Thread(target=eeg_reader, daemon=True)
        eeg_thread.start()

    emit(
        "eeg_status",
        {"running": True, "message": "Starting EEG process..."},
    )


@socketio.on("stop_live")
def handle_stop_live():
    """Stop the EEG subprocess."""
    global eeg_process, eeg_running

    with eeg_lock:
        if eeg_process and eeg_running:
            try:
                eeg_process.terminate()
            except Exception:
                pass
        else:
            emit(
                "eeg_status",
                {"running": False, "message": "EEG was not running."},
            )
            return

    emit(
        "eeg_status",
        {"running": False, "message": "Stopping EEG process..."},
    )


if __name__ == "__main__":
    # Run on localhost:5000 with Socket.IO
    socketio.run(app, debug=True)
