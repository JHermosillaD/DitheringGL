# Real-Time Dithering Effect

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![ModernGL](https://img.shields.io/badge/ModernGL-5.12+-lightgrey?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)
![OS](https://img.shields.io/badge/OS-Linux-orange?style=flat-square&logo=linux&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)

A real-time project built with ModernGL. It isolates the person from the background, obscures facial details and runs a custom OpenGL shader to generate a cyberpunk-inspired dithering effect.

<video width="480" height="854" src="https://github.com/user-attachments/assets/53e04a60-3c2c-4c3e-8e12-1f49927e5414"></video>

### Project Structure
```text
├── main.py
├── app.py
├── shaders/
│   ├── passthrough.vert
│   └── passthrough.frag
└── models/
    └── selfie_segmenter_landscape.tflite
```

### Credits

* Segmentation Model by Google.
