# Start with default settings (camera 0, 1280x720)
python main.py

# Use specific camera
python main.py --camera 1

# Change resolution
python main.py --width 1920 --height 1080

# Specific detection mode
python main.py --mode faces    # Face detection only
python main.py --mode objects  # Object detection only
python main.py --mode both     # Both (default)
