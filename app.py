from flask import Flask, render_template, request, jsonify
import csv
import ast
import os
import wave
import contextlib

app = Flask(__name__)

# For now we hardcode available CSVs and songs.
# Later you can expand these lists or auto-discover files in a folder.
CSV_FILES = {
    "Sample Muse Stream": "muse_stream.csv"
}

SONG_FILES = {
    "Sample Song": "music.wav"
}


def load_eeg_from_csv(csv_path):
    """
    Reads the CSV and extracts 4 EEG channels from /muse/eeg rows.
    Assumes column layout: [timestamp, address, values]
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
            # Map channels:
            # ch1 = left temporal
            # ch2 = left frontal
            # ch3 = right frontal
            # ch4 = right temporal
            ch1.append(float(values[0]))
            ch2.append(float(values[1]))
            ch3.append(float(values[2]))
            ch4.append(float(values[3]))

        # Decide if first_row is header or data
        if "muse" in "".join(first_row).lower():
            # Probably data
            process_row(first_row)
        else:
            # Probably header – do nothing, continue
            pass

        for row in reader:
            process_row(row)

    return ch1, ch2, ch3, ch4


def get_wav_duration(path):
    """
    Returns duration of a WAV file in seconds, or None if unavailable.
    """
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
    Receives selected csv + song from the frontend, parses the CSV,
    and returns the EEG time series as JSON for the browser to animate.
    Time axis is stretched/compressed to match the song duration.
    """
    data = request.get_json()
    csv_key = data.get("csv_file")
    song_key = data.get("song_file")

    # Resolve file names
    csv_filename = CSV_FILES.get(csv_key)
    song_filename = SONG_FILES.get(song_key)

    if not csv_filename:
        return jsonify({"error": "Invalid CSV selection"}), 400

    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, csv_filename)
    song_path = os.path.join(base_dir, song_filename) if song_filename else None

    try:
        ch1, ch2, ch3, ch4 = load_eeg_from_csv(csv_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    n = len(ch1)
    if n == 0:
        # No data found, but return 200 so frontend can handle gracefully
        return jsonify({"error": "No /muse/eeg data found in CSV"}), 200

    # Compute time axis based on song duration
    song_duration = get_wav_duration(song_path) if song_path else None

    if song_duration and n > 1:
        # Spread samples evenly across the song's duration
        times = [i * song_duration / (n - 1) for i in range(n)]
    else:
        # Fallback: just use sample index if duration unknown
        times = list(range(n))
        song_duration = None  # indicate we didn't actually sync

    return jsonify({
        "times": times,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "song": song_filename,
        "song_duration": song_duration
    })


if __name__ == "__main__":
    app.run(debug=True)
