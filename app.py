from flask import Flask, request, jsonify
import os
from transformers import pipeline
import librosa
import soundfile as sf
import tempfile
import cloudinary
import cloudinary.uploader
import requests

app = Flask(__name__)

def configure_cloudinary():
    cloudinary.config(
        cloud_name="hcvu40dvj",
        api_key="523379171599888",
        api_secret="pD2VU84Ew_KMETn0o-6kdbjPFnU"
    )

configure_cloudinary()

# Function to load an audio file and resample it
def load_and_resample_audio(file_path, target_sr=48000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, sr

# Function to initialize the zero-shot audio classification pipeline
def initialize_zero_shot_pipeline(model_name):
    return pipeline(task="zero-shot-audio-classification", model=model_name)

# Function to classify an audio sample
def classify_audio_sample(classifier, audio_sample, candidate_labels):
    return classifier(audio_sample, candidate_labels=candidate_labels)

# Function to detect classes and their timestamps in the audio
def detect_classes(audio, sampling_rate, classifier, candidate_labels):
    segment_length = 5  # seconds
    detected_classes = []

    # Segment the audio
    for i in range(0, len(audio), segment_length * sampling_rate):
        segment = audio[i:i + segment_length * sampling_rate]

        # Classify the segment
        classification_result = classify_audio_sample(classifier, segment, candidate_labels)

        # Record detected classes and their timestamps
        detected_classes.append({
            "start_time": i / sampling_rate,
            "end_time": min((i + segment_length * sampling_rate) / sampling_rate, len(audio) / sampling_rate),
            "detected_classes": classification_result[0]['label']
        })

    return detected_classes

# Customize candidate labels for cheating detection
candidate_labels = [
    "Sneezing", "Clapping", "Breathing", "Coughing", "Laughing", 
    "Snoring", "Drinking, sipping", "Mouse click", "Keyboard typing", "No sound", "Talking"
]

# Customize model name
model_name = "laion/clap-htsat-unfused"  # You can change the model here
zero_shot_classifier = initialize_zero_shot_pipeline(model_name)

@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Upload file to Cloudinary
    upload_result = cloudinary.uploader.upload(file.stream, resource_type="auto")

    # Get the public URL of the uploaded file
    file_url = upload_result['secure_url']

    # Download the file to a temporary location
    temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_url))
    download_file(file_url, temp_file_path)

    # Load and resample the audio file
    audio_sample, sampling_rate = load_and_resample_audio(temp_file_path)

    # Detect classes and their timestamps in the audio
    detected_classes = detect_classes(audio_sample, sampling_rate, zero_shot_classifier, candidate_labels)

    # Remove the temporary file after processing
    os.remove(temp_file_path)

    # Return the response with file link and cheating details
    response_data = {
        "file": file_url,
        "cheatingDetails": "taking"
    }

    return jsonify(response_data)

def download_file(url, file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
