# main.py
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import 
import argparse
from collections import deque
from datetime import datetime

class RealTimeComputerVision:
    """Main class for real-time computer vision system"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Initialize MediaPipe Face Detection
        self.initialize_mediapipe()
        
        # Initialize Object Detection
        self.initialize_object_detection()
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection results storage
        self.faces = []
        self.objects = []
        
        # Detection modes
        self.mode = self.config.get('mode', 'both')
        
        # Colors for visualization
        self.colors = {
            'face': (0, 255, 0),      # Green for faces
            'person': (255, 0, 0),    # Blue for person
            'vehicle': (0, 0, 255),   # Red for vehicles
            'default': (255, 255, 0)  # Cyan for other objects
        }
        
        # Stats
        self.stats = {
            'total_faces': 0,
            'total_objects': 0,
            'processing_time': 0
        }
    
    def initialize_mediapipe(self):
        """Initialize MediaPipe face detection"""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            print("✓ MediaPipe initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing MediaPipe: {e}")
            sys.exit(1)
    
    def initialize_object_detection(self):
        """Initialize object detection model"""
        try:
            # Check if model files exist
            if not self.check_model_files():
                print("Model files not found. Attempting to download...")
                self.download_model_files()
            
            # Load the model
            model_config = self.get_model_config()
            self.net = cv2.dnn.readNetFromCaffe(
                model_config['prototxt'],
                model_config['model']
            )
            
            # Set preferable backend if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("✓ Using CUDA acceleration")
            
            self.CLASSES = model_config['classes']
            print(f"✓ Object detection model loaded ({len(self.CLASSES)} classes)")
            
        except Exception as e:
            print(f"✗ Error initializing object detection: {e}")
            self.net = None
    
    def get_model_config(self):
        """Get model configuration"""
        return {
            'prototxt': 'models/MobileNetSSD_deploy.prototxt',
            'model': 'models/MobileNetSSD_deploy.caffemodel',
            'classes': [
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"
            ]
        }
    
    def check_model_files(self):
        """Check if model files exist"""
        config = self.get_model_config()
        return os.path.exists(config['prototxt']) and os.path.exists(config['model'])
    
    def download_model_files(self):
        """Download model files if missing"""
        import urllib.request
        import ssl
        
        # Create a default SSL context to handle certificate issues
        ssl._create_default_https_context = ssl._create_unverified_context
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        config = self.get_model_config()
        
        # URLs for the model files
        urls = {
            'prototxt': "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
            # For the model file, we'll use an alternative source since Google Drive requires authentication
        }
        
        print("Downloading model files...")
        
        # Download prototxt
        if not os.path.exists(config['prototxt']):
            try:
                print(f"  Downloading prototxt...")
                urllib.request.urlretrieve(urls['prototxt'], config['prototxt'])
                print(f"  ✓ Downloaded prototxt")
            except Exception as e:
                print(f"  ✗ Failed to download prototxt: {e}")
                return False
        
        # For the model file, we'll use a direct link or provide instructions
        if not os.path.exists(config['model']):
            print("\n" + "="*60)
            print("⚠️  MobileNet SSD model file required!")
            print("="*60)
            print("Please download the model manually:")
            print("1. Visit: https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view")
            print("2. Click 'Download' (you need to be signed into Google)")
            print(f"3. Save it as: {config['model']}")
            print("\nAlternative download using wget:")
            print(f"wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc' -O {config['model']}")
            print("="*60)
            return False
        
        return True
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.mode not in ['both', 'faces']:
            return []
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.face_detection.process(frame_rgb)
        
        detected_faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                confidence = detection.score[0]
                
                if confidence > 0.5:
                    detected_faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': float(confidence),
                        'label': 'Face'
                    })
        
        return detected_faces
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.mode not in ['both', 'objects'] or self.net is None:
            return []
        
        (h, w) = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        
        # Perform detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected_objects = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = self.CLASSES[idx]
                
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are valid
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                width = endX - startX
                height = endY - startY
                
                detected_objects.append({
                    'label': label,
                    'bbox': (startX, startY, width, height),
                    'confidence': float(confidence)
                })
        
        return detected_objects
    
    def process_frame(self, frame):
        """Process a single frame"""
        start_time = time.time()
        
        # Detect faces and objects
        self.faces = self.detect_faces(frame)
        self.objects = self.detect_objects(frame)
        
        # Update stats
        self.stats['processing_time'] = (time.time() - start_time) * 1000  # ms
        self.stats['total_faces'] += len(self.faces)
        self.stats['total_objects'] += len(self.objects)
        
        return frame
    
    def draw_detections(self, frame):
        """Draw detections on frame"""
        # Draw face detections
        for face in self.faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), 
                         self.colors['face'], 2)
            
            # Draw label
            label = f"Face: {confidence:.2f}"
            self.draw_label(frame, label, (x, y - 10), 
                           self.colors['face'])
        
        # Draw object detections
        for obj in self.objects:
            x, y, w, h = obj['bbox']
            label = obj['label']
            confidence = obj['confidence']
            
            # Choose color based on object type
            color = self.colors.get(label.lower(), self.colors['default'])
            if label == 'person':
                color = self.colors['person']
            elif label in ['car', 'bus', 'motorbike', 'bicycle']:
                color = self.colors['vehicle']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            obj_label = f"{label}: {confidence:.2f}"
            self.draw_label(frame, obj_label, (x, y - 10), color)
        
        return frame
    
    def draw_label(self, frame, text, position, color):
        """Draw a label with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(frame, 
                     (x, y - text_height - 10),
                     (x + text_width, y),
                     color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    def draw_stats_panel(self, frame):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Display information
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Mode: {self.mode}",
            f"Time: {current_time}",
            f"Faces: {len(self.faces)}",
            f"Objects: {len(self.objects)}",
            f"Process: {self.stats['processing_time']:.1f}ms",
            f"Total Faces: {self.stats['total_faces']}",
            f"Total Objects: {self.stats['total_objects']}"
        ]
        
        y_offset = 40
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def draw_controls(self, frame):
        """Draw controls information"""
        h, w = frame.shape[:2]
        
        controls = [
            "CONTROLS:",
            "F - Face detection only",
            "O - Object detection only",
            "B - Both detections",
            "S - Save screenshot",
            "R - Reset statistics",
            "Q - Quit application"
        ]
        
        y_offset = h - 150
        for i, control in enumerate(controls):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, control, (w - 300, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        return frame
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Screenshot saved: {filename}")
        return filename
    
    def run(self, camera_index=0, width=1280, height=720):
        """Main run loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("✗ Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("Real-Time Computer Vision System")
        print("="*50)
        print("Controls:")
        print("  F - Face detection only")
        print("  O - Object detection only")
        print("  B - Both detections")
        print("  S - Save screenshot")
        print("  R - Reset statistics")
        print("  Q - Quit application")
        print("="*50)
        
        print(f"\nStarting camera {camera_index} at {width}x{height}")
        print("Press 'Q' to exit...\n")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Failed to capture frame")
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Draw visualizations
            frame = self.draw_detections(frame)
            frame = self.draw_stats_panel(frame)
            frame = self.draw_controls(frame)
            
            # Update FPS
            self.update_fps()
            
            # Display frame
            cv2.imshow('Real-Time Computer Vision', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.mode = 'faces'
                print("✓ Mode: Face detection only")
            elif key == ord('o'):
                self.mode = 'objects'
                print("✓ Mode: Object detection only")
            elif key == ord('b'):
                self.mode = 'both'
                print("✓ Mode: Both detections")
            elif key == ord('s'):
                self.save_screenshot(frame)
            elif key == ord('r'):
                self.stats = {
                    'total_faces': 0,
                    'total_objects': 0,
                    'processing_time': 0
                }
                print("✓ Statistics reset")
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                pass
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                pass
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        
        print("\n" + "="*50)
        print("Session Summary:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Average FPS: {self.fps:.1f}")
        print(f"  Total faces detected: {self.stats['total_faces']}")
        print(f"  Total objects detected: {self.stats['total_objects']}")
        print("="*50)
        print("Application closed.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Real-Time Computer Vision System with Face and Object Detection'
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['both', 'faces', 'objects'],
                       help='Detection mode (default: both)')
    
    args = parser.parse_args()
    
    # Create and run the vision system
    config = {
        'mode': args.mode
    }
    
    vision_system = RealTimeComputerVision(config)
    vision_system.run(
        camera_index=args.camera,
        width=args.width,
        height=args.height
    )

if __name__ == "__main__":
    main()
