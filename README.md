# 📸 Visionary AI - Image Caption Generator

Visionary AI is a state-of-the-art deep learning project that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to automatically generate descriptive captions for images.

## 🚀 Features
- **CNN-LSTM Architecture**: Uses ResNet50 for image feature extraction and LSTM for sequence generation.
- **Premium Web Interface**: Sleek, glassmorphism-based UI for uploading images and viewing generated captions.
- **Voice Output**: Integrated text-to-speech for both web and CLI.
- **Easy Training**: Includes a comprehensive training pipeline for datasets like Flickr8k.

## 🛠️ Tech Stack
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, PIL
- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS (Modern & Aesthetic)

## 📁 Project Structure
- `train.py`: Main training pipeline.
- `predict.py`: Standalone CLI prediction script.
- `server/main.py`: FastAPI server for the web interface.
- `web/`: Premium frontend assets (HTML, CSS, JS).
- `utils/`: Data loading and model building utilities.
- `model/`: Directory for saved model weights (`.h5`).

## ⚙️ Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web App**:
   - Start the FastAPI server:
     ```bash
     python server/main.py
     ```
   - Open `web/index.html` in your browser.

3. **Run CLI Prediction**:
   ```bash
   python predict.py --image path/to/your/image.jpg --voice
   ```

## 🧠 Model Workflow
1. **Input**: Image is resized to 224x224.
2. **CNN**: ResNet50 extracts a 2048-dimensional feature vector.
3. **LSTM**: Sequence generator predicts the caption word-by-word, starting with `startseq` and ending with `endseq`.
4. **Output**: A natural language sentence describing the visual content.

---
Built with ❤️ by Antigravity AI
