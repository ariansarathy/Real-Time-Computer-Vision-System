## Real-Time Multi-Modal Computer Vision System

A production-ready real-time computer vision pipeline that integrates **MediaPipe Face Detection** and **MobileNet SSD Object Detection** using OpenCV. This system performs simultaneous face and object detection with sub-50ms inference times, customizable detection modes, and comprehensive performance monitoring.

# Real-Time Computer Vision System

Real-time face and object detection system using OpenCV and MediaPipe.

**Key Features:**
- **Dual-Model Architecture**: Parallel processing with MediaPipe (faces) and MobileNet SSD (80 object classes)
- **Real-Time Performance**: Optimized for 30+ FPS processing on standard hardware
- **Multiple Detection Modes**: Toggle between face-only, object-only, or combined detection
- **Comprehensive Visualization**: Color-coded bounding boxes, confidence scores, and real-time statistics
- **Model Auto-Download**: Automatic retrieval of pre-trained models on first run

**Technical Highlights:**
- MediaPipe's BlazeFace for high-speed face detection
- OpenCV DNN module with Caffe-based MobileNet SSD
- CUDA acceleration support for GPU-enabled systems
- Command-line interface with configurable parameters
- Performance analytics and screenshot capabilities
