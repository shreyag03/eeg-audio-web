from flask import Flask, render_template, request, jsonify
import csv
import ast
import os
import wave
import contextlib

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(__file__)
STREAM_CSV_FILENAME = "muse_stream.csv"

# We hard-code available CSVs and songs.
# "Live Stream (Muse Headband)" is handled specially on the frontend.
CSV_FILES = {
    "Sample Muse Stream": "muse_stream.csv",
    "Live Stream (Muse Headband)": "STREAM"  # sentinel, not used by /visualize
}

SONG_FILES = {
    "Sample Song": "music.wav",
    "No song (EEG only)": "NONE"
}

# How far we've read into the streaming CSV file
_stream_file_offset = 0


def load_eeg_from_csv(csv_path):
    """
    Reads the CSV and extracts 4 EEG channels from /muse/eeg rows.
    Assumes columns: [timestamp, address, values]
    where values is a string like "[v1, v2, v3, v4]".
    """
    ch1, ch2, ch3, ch4 = [], [], [], []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return [], [], [], []

        def process_row(row):
            if len(row) < 3:
                return
            address = row[1].strip()
            if address != "/muse/eeg":
                return
            values_str = row[2]
            try:
                values = ast.literal_eval(values_str)
            except Exception:
                return
            if not isinstance(values, (list, tuple)) or len(values) < 4:
                return
            ch1.append(float(values[0]))
            ch2.append(float(values[1]))
            ch3.append(float(values[2]))
            ch4.append(float(values[3]))

        # Decide if first_row is header or data
        if "muse" in "".join(first_row).lower():
            process_row(first_row)
        else:
            # Probably header – do nothing
            pass

        for row in reader:
            process_row(row)

    return ch1, ch2, ch3, ch4


def get_wav_duration(path):
    """Returns duration of a WAV file in seconds, or None if unavailable."""
    if not path or not os.path.exists(path):
        return None
    try:
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        csv_files=CSV_FILES,
        song_files=SONG_FILES
    )


@app.route("/visualize", methods=["POST"])
def visualize():
    """
    File-based visualization:
    Receives selected csv + song (offline mode), parses the CSV,
    and returns the EEG time series as JSON for the browser to animate.
    Time axis is stretched/compressed to match the song duration.
    """
    data = request.get_json()
    csv_key = data.get("csv_file")
    song_key = data.get("song_file")

    # Live stream is handled by /stream_* endpoints, not here.
    if csv_key == "Live Stream (Muse Headband)":
        return jsonify({"error": "Live stream is handled via streaming endpoints."}), 400

    csv_filename = CSV_FILES.get(csv_key)
    song_filename = SONG_FILES.get(song_key)

    if not csv_filename or csv_filename == "STREAM":
        return jsonify({"error": "Invalid CSV selection"}), 400

    csv_path = os.path.join(BASE_DIR, csv_filename)
    song_path = os.path.join(BASE_DIR, song_filename) if song_filename and song_filename != "NONE" else None

    try:
        ch1, ch2, ch3, ch4 = load_eeg_from_csv(csv_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    n = len(ch1)
    if n == 0:
        return jsonify({"error": "No /muse/eeg data found in CSV"}), 200

    # Compute time axis based on song duration if available
    song_duration = get_wav_duration(song_path) if song_path else None

    if song_duration and n > 1:
        times = [i * song_duration / (n - 1) for i in range(n)]
    else:
        times = list(range(n))
        song_duration = None

    return jsonify({
        "times": times,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "song": song_filename,
        "song_duration": song_duration
    })


@app.route("/stream_reset", methods=["POST"])
def stream_reset():
    """
    Reset streaming offset so that a new live session starts reading
    from the beginning of muse_stream.csv.
    The frontend calls this just before starting live visualization.
    """
    global _stream_file_offset
    _stream_file_offset = 0
    return jsonify({"status": "reset"})


@app.route("/stream_samples")
def stream_samples():
    """
    Streaming endpoint:
    Reads NEW /muse/eeg rows appended to muse_stream.csv since the last call.
    Returns only incremental samples, suitable for real-time plotting.

    Expected file structure (from ece446.py):
        timestamp,address,values
        2025-..,/muse/eeg,"[v1, v2, v3, v4]"
        ...
    """
    global _stream_file_offset

    csv_path = os.path.join(BASE_DIR, STREAM_CSV_FILENAME)
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Streaming CSV not found. Is ece446.py running?",
            "ch1": [],
            "ch2": [],
            "ch3": [],
            "ch4": [],
            "raw_lines": []
        }), 200

    ch1, ch2, ch3, ch4 = [], [], [], []
    raw_lines = []

    # Handle file truncation (e.g., new stream overwrote the file)
    current_size = os.path.getsize(csv_path)
    if _stream_file_offset > current_size:
        _stream_file_offset = 0

    with open(csv_path, "r", newline="") as f:
        # Seek to last read position
        f.seek(_stream_file_offset)
        reader = csv.reader(f)

        def process_row(row):
            if len(row) < 3:
                return
            address = row[1].strip()
            if address != "/muse/eeg":
                return
            values_str = row[2]
            try:
                values = ast.literal_eval(values_str)
            except Exception:
                return
            if not isinstance(values, (list, tuple)) or len(values) < 4:
                return
            ch1.append(float(values[0]))
            ch2.append(float(values[1]))
            ch3.append(float(values[2]))
            ch4.append(float(values[3]))
            raw_lines.append(",".join(row))

        # If we are at the beginning, skip header if present
        if _stream_file_offset == 0:
            first_row = next(reader, None)
            if first_row is not None:
                joined = ",".join(first_row).lower()
                if "timestamp" not in joined or "address" not in joined:
                    # Not a header – treat it as data
                    process_row(first_row)

        for row in reader:
            process_row(row)

        # Update offset for next call
        _stream_file_offset = f.tell()

    return jsonify({
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "raw_lines": raw_lines
    })


if __name__ == "__main__":
    app.run(debug=True)
