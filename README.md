# Real-Time Computer Vision System

Real-time face and object detection system using OpenCV and MediaPipe.

## ✨ Features
- **Real-time face detection** using MediaPipe's optimized models
- **Real-time object detection** with MobileNet SSD (80+ object classes)
- **Multiple detection modes**: Face-only, object-only, or both simultaneously
- **Performance statistics**: Real-time FPS, processing time, detection counts
- **Screenshot capture**: Save detection results with timestamp
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux
- **CUDA acceleration**: GPU support for faster processing
- **Clean visualization**: Professional UI with stats overlay

**Technical Highlights:**
- MediaPipe's BlazeFace for high-speed face detection
- OpenCV DNN module with Caffe-based MobileNet SSD
- CUDA acceleration support for GPU-enabled systems
- Command-line interface with configurable parameters
- Performance analytics and screenshot capabilities


## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or camera source

### Quick Start
```bash
# Clone the repository
git clone https://github.com/ariansarathy/real-time-cv-system.git
cd real-time-cv-system

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
