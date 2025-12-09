import cv2
import numpy as np
import keras
import gradio as gr
import tempfile
import os
import time

# Load model (HF safe)
model = keras.models.load_model("model_file.h5", compile=False)

# Load Haar cascade
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if cascade.empty():
    raise RuntimeError("❌ Haar cascade file NOT FOUND in Space!")

labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def process_video(video_file):
    time.sleep(1)

    # Handle Gradio upload
    if isinstance(video_file, dict) and "name" in video_file:
        video_file = video_file["name"]

    if not os.path.exists(video_file):
        return "❌ Uploaded video path invalid."

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "❌ Cannot open uploaded video"

    ret, frame = cap.read()
    if not ret:
        return "❌ Cannot read video frames"

    h, w = frame.shape[:2]

    # Create output temp file (.webm because mp4 encoding not supported!)
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.webm")

    # VP9 codec works in HuggingFace
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 3)

        for (x, y, wf, hf) in faces:
            face = gray[y:y+hf, x:x+wf]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0

            pred = model.predict(face, verbose=0)
            emotion = labels[int(np.argmax(pred))]

            cv2.rectangle(frame, (x, y), (x+wf, y+hf), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()

    return output_path


interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload classroom/student video"),
    outputs=gr.Video(label="Emotion Detection Result"),
    title="Student Facial Emotion Recognition",
    description="Upload a classroom or student video to analyze emotions."
)

interface.launch()
