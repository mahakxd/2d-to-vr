import os
import subprocess
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import google.genai as genai

# ---------------- GEMINI CONFIG ----------------
API_KEY = os.environ.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=API_KEY)

# ---------------- EFFECT FUNCTIONS ----------------
def create_fisheye(frame, strength=0.5):
    # simplified fisheye effect for demo
    h, w = frame.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = w/2, h/2
    x_norm, y_norm = (x-cx)/(w/2), (y-cy)/(h/2)
    r = np.sqrt(x_norm**2 + y_norm**2)
    mask = r <= 1
    theta = np.arctan2(y_norm, x_norm)
    r_dist = r * strength
    new_x = cx + (w/2) * r_dist * np.cos(theta)
    new_y = cy + (h/2) * r_dist * np.sin(theta)
    map_x, map_y = x.copy(), y.copy()
    map_x[mask] = new_x[mask]
    map_y[mask] = new_y[mask]
    return cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

def create_spherical(frame, strength=0.7):
    h, w = frame.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = w/2, h/2
    dx, dy = x-cx, y-cy
    r = np.sqrt(dx**2 + dy**2)
    mask = r < min(cx, cy)*0.8
    theta = np.arctan2(dy, dx)
    phi = np.pi * r / (min(cx, cy)*0.8) * strength
    new_x = cx + (min(cx, cy)*0.8) * np.sin(phi) * np.cos(theta)
    new_y = cy + (min(cx, cy)*0.8) * np.sin(phi) * np.sin(theta)
    map_x, map_y = x.copy(), y.copy()
    map_x[mask] = new_x[mask]
    map_y[mask] = new_y[mask]
    return cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

def create_equirect(frame, strength=0.6):
    h, w = frame.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = w/2, h/2
    norm_x, norm_y = (x-cx)/cx, (y-cy)/cy
    lon = norm_x * np.pi * strength
    lat = norm_y * np.pi/2 * strength
    new_x = ((lon/np.pi + 1)*cx).astype(np.float32)
    new_y = ((lat/(np.pi/2)+1)*cy).astype(np.float32)
    new_x = np.clip(new_x, 0, w-1)
    new_y = np.clip(new_y, 0, h-1)
    return cv2.remap(frame, new_x, new_y, cv2.INTER_LINEAR)

def create_stereo_frame(frame, effect_func, strength=0.7, eye_offset=10):
    left = effect_func(frame, strength)
    right = effect_func(frame, strength)
    h, w = left.shape[:2]
    # shift right eye for VR
    M = np.float32([[1,0,eye_offset],[0,1,0]])
    right_shifted = cv2.warpAffine(right, M, (w, h))
    return np.concatenate([left, right_shifted], axis=1)

# ---------------- VIDEO PROCESSING ----------------
def process_video(video_path, style="equirect", strength=0.7, eye_offset=10, volume=1.0):
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name
    temp_output.close()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    if style == "fisheye":
        effect_func = create_fisheye
    elif style == "spherical":
        effect_func = create_spherical
    else:
        effect_func = create_equirect

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stereo = create_stereo_frame(frame, effect_func, strength, eye_offset)
        out.write(stereo)
        frame_count += 1

    cap.release()
    out.release()

    # Adjust audio volume using ffmpeg
    vol_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = ["ffmpeg","-i",output_path,"-filter:a",f"volume={volume}","-c:v","copy",vol_output,"-y","-loglevel","quiet"]
    subprocess.run(cmd)

    return vol_output, f"✅ Processed {frame_count} frames with {style} VR effect!"

# ---------------- GEMINI AUTO STYLE ----------------
def suggest_style(video_description):
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"text": f"Suggest best VR conversion style (fisheye, spherical, equirectangular) for this video: {video_description}"}]
        )
        return response.text.strip().lower()
    except:
        return "equirect"

# ---------------- GRADIO UI ----------------
with gr.Blocks(title="2D → VR Video Converter") as demo:
    gr.Markdown("## 🎥 Convert 2D Videos to Side-by-Side VR")
    
    with gr.Tabs():
        with gr.Tab("Converter"):
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(label="Upload Video")
                    description = gr.Textbox(label="Video Description (optional)", placeholder="Describe your video for auto style...")
                    auto_style_btn = gr.Button("Auto Suggest Style")
                    style_choice = gr.Radio(["equirect", "fisheye", "spherical"], label="VR Style", value="equirect")
                    strength_slider = gr.Slider(0.3, 1.0, value=0.7, step=0.1, label="Effect Strength")
                    eye_offset_slider = gr.Slider(5,50,value=10,step=1,label="VR Eye Offset (pixels)")
                    volume_slider = gr.Slider(0.0,2.0,value=1.0,step=0.1,label="Volume Multiplier")
                    convert_btn = gr.Button("Convert to VR")
                with gr.Column():
                    output_video = gr.Video(label="VR Side-by-Side Output")
                    status_box = gr.Textbox(label="Status")
        
    def run_auto_style(desc):
        if not desc.strip():
            return "equirect"
        return suggest_style(desc)
    
    def run(video, desc, style, strength, eye_offset, volume):
        if not style:
            style = run_auto_style(desc)
        return process_video(video, style, strength, eye_offset, volume)
    
    auto_style_btn.click(run_auto_style, inputs=description, outputs=style_choice)
    convert_btn.click(run, inputs=[video_in, description, style_choice, strength_slider, eye_offset_slider, volume_slider],
                      outputs=[output_video, status_box])

demo.launch()
