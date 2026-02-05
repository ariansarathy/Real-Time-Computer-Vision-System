# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="real-time-cv-system",
    version="1.0.0",
    author="Arian Sarathy",
    author_email="arian.sarathy@gmail.com",
    description="Real-time face and object detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ariansarathy/Real-Time-Computer-Vision-System",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        "console_scripts": [
            "cv-system=main:main",
        ],
    },
)
