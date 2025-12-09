# student-facial-emotion-regression
Deep Learning-based Student Facial Emotion Recognition with video input, HuggingFace demo, and OpenCV  



🎓 Student Facial Emotion Recognition

A simple and effective deep-learning project that identifies student emotions from classroom videos.
The model analyzes each detected face in the video and predicts one of seven emotions in real time.

👉 Live Demo:
https://huggingface.co/spaces/vikneshveluchamy/student-emotion-regression

📌 Overview

This project was created to understand how computer vision and deep learning can be used to study student engagement and classroom behavior.
Given any recorded classroom video, the system:

detects faces using Haar Cascade

preprocesses each face into 48×48 grayscale

predicts emotion using a trained CNN model

writes the emotion label back on the video

outputs a processed video

Everything runs on CPU, so it works on laptops and free-tier cloud environments.

🧠 Emotions Detected

The model predicts the following seven emotion classes:

😊 Happy

😐 Neutral

😠 Angry

😢 Sad

😨 Fear

😮 Surprise

😖 Disgust

🚀 Live Interactive Demo

You can try the project directly in your browser:

🔗 HuggingFace Space:
https://huggingface.co/spaces/vikneshveluchamy/student-emotion-regression

Upload any classroom or student video and the app will return an annotated version with predicted emotions.

🗂 Project Structure
student-facial-emotion-regression/
│
├── app.py                          # Gradio app for HuggingFace
├── predict.py                      # Local video inference script
├── test.py                         # Evaluation / testing script
├── haarcascade_frontalface_default.xml
├── model_file.h5                   # Trained emotion recognition model
├── requirements.txt
└── README.md

⚙️ How It Works (Short Explanation)

Face Detection
OpenCV’s Haar Cascade is used to locate faces in each frame.

Preprocessing

Cropped face

Resized to 48×48

Converted to grayscale

Normalized (0–1)

Emotion Prediction
A CNN model trained on FER2013 outputs probabilities for each emotion.

Video Output
The predicted emotion is written on top of the face and saved as a new video.

▶️ Running the Project Locally

Install the required packages:

pip install -r requirements.txt


Run prediction on any video:

python predict.py


Make sure the following files are in the same folder:

predict.py

model_file.h5

haarcascade_frontalface_default.xml

📦 Model Used

The project uses a custom-trained Keras CNN model with the following layers:

multiple Conv2D + ReLU

MaxPooling

Dropout for regularization

Dense layers for classification

Softmax output layer

Training dataset: FER2013


🔧 Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Gradio (for web demo)

HuggingFace Spaces

🎯 Why This Project Matters

This project demonstrates how AI can help in:

analyzing classroom engagement

understanding student behavior

assisting teachers with feedback

building educational analytics tools

🤝 Contributions

Feel free to submit improvements or suggestions.
Pull requests are always welcome.

📜 License

MIT License
