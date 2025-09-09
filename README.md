
2D → VR Video Converter

🎥 Convert regular 2D videos into immersive VR-style experiences!

This project allows you to take any standard video and transform it into a VR-compatible format with side-by-side stereoscopic output for VR headsets. You can choose different VR effects (equirectangular, fisheye, spherical), control effect strength, and adjust audio levels.

The app uses Gradio for an interactive web interface and leverages OpenCV, NumPy, Pillow, and Google Gemini API for optional AI-powered enhancements.

Features
>Upload your 2D video and convert it to VR side-by-side format
>Select VR effect style: equirectangular, fisheye, or spherical
>Control the strength of the VR effect
>Adjust audio volume
>Generates a depth map preview of the scene
>Runs entirely in your browser via Gradio
>Optimized processing for smooth playback

Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/mahak0812/2d-to-vr-video.git
cd 2d-to-vr-video
pip install -r requirements.txt
```

> Make sure you have a valid Google Gemini API key for AI-enhanced features.

---

Usage

Run the Gradio interface locally:

```bash
python app.py
```
>Upload your video, select the VR style and effect strength
>Click Convert to VR
>View/download the processed VR video and depth map

Example

| Upload Video                  | Side-by-Side VR Output           |
| ----------------------------- | -------------------------------- |
| ![upload](example_upload.png) | ![output](example_vr_output.png) |

Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the VR conversion pipeline, add new VR effects, or optimize performance.
