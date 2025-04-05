import os
import tempfile
import uuid
from typing import List

import moviepy.config as mpy_config
import numpy as np
import scipy.io.wavfile as wav
import torch
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

from langchain_wrappers.audio_generation import generate_audio

mpy_config.change_settings({
    "IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
})


def create_video_from_images_and_audio(
        images: List,
        audio_tensors: List,
        sample_rate: int = 22050,
        poem_lines: List[str] = None,
        poet: str = None,
        poem_name: str = ""
) -> str:
    """
    Create a video from lists of PIL images and audio tensors with an introductory title screen.
    Returns the output video file path.
    """
    if len(images) != len(audio_tensors):
        raise ValueError("Number of images must match number of audio tensors")

    # Create a temporary directory for audio files.
    temp_dir = tempfile.mkdtemp()
    final_clip = None
    video_clips = []
    audio_paths = []
    intro_audio_path = None
    try:
        # Generate introductory audio for the title screen.
        intro_text = f"{poem_name} by {poet}" if poet else poem_name
        # Use your audio generation function
        intro_audio = generate_audio(intro_text)
        intro_audio_tensor = torch.tensor(intro_audio, dtype=torch.float32)
        intro_audio_path = os.path.join(temp_dir, "intro_audio.wav")
        intro_audio_array = intro_audio_tensor.cpu().numpy()
        intro_audio_array = (intro_audio_array * 32767).astype(np.int16)
        wav.write(intro_audio_path, sample_rate, intro_audio_array)

        intro_audio_clip = AudioFileClip(intro_audio_path)
        intro_duration = intro_audio_clip.duration

        # Create title clip with antialiasing and higher resolution
        title_clip = TextClip(
            poem_name,
            fontsize=80,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center',
            size=(1100, None)
        ).set_position(("center", 100)).set_duration(intro_duration)

        # Create poet clip with improved rendering
        poet_clip = TextClip(
            f"by {poet}" if poet else "",
            fontsize=40,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center'
        ).set_position(("center", 100 + title_clip.h + 20)).set_duration(intro_duration)

        # Create the title screen with just the single combined clip
        title_screen = CompositeVideoClip([
            ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8)).set_duration(intro_duration),
            title_clip,
            poet_clip
        ]).set_audio(intro_audio_clip)

        video_clips.append(title_screen)

        # Convert each audio tensor to a WAV file.
        for i, audio_tensor in enumerate(audio_tensors):
            temp_audio_path = os.path.join(temp_dir, f"temp_audio_{i}.wav")
            audio_array = audio_tensor.cpu().numpy()
            audio_array = (audio_array * 32767).astype(np.int16)
            wav.write(temp_audio_path, sample_rate, audio_array)
            audio_paths.append(temp_audio_path)

        print(f"{len(audio_paths)=}")
        print(f"{len(images)=}")
        print(f"Poem lines: {poem_lines}")
        print(f"Number of images: {len(images)}")
        print(f"Number of poem lines: {len(poem_lines) if poem_lines else 0}")

        # Create a video clip for each image and corresponding audio.
        for idx, (img, audio_path) in enumerate(zip(images, audio_paths)):
            # Ensure the image is in RGB mode and resize to standard dimensions if needed
            img = img.convert("RGB")
            # Resize image to a standard size that works well with video (e.g., 1280x720)
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # Verify the image array has proper shape and values
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                print(f"Warning: Image at index {idx} has unexpected shape: {img_array.shape}")
                # Create a blank image instead of failing
                img_array = np.zeros((720, 1280, 3), dtype=np.uint8)

            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            image_clip = ImageClip(img_array).set_duration(audio_duration).set_fps(24)

            if poem_lines and idx % 2 == 0 and idx // 2 < len(poem_lines):
                max_width = int(img_array.shape[1] * 0.9)
                text_line = poem_lines[idx // 2].upper()
                text_clip = TextClip(
                    text_line,
                    fontsize=60,
                    color='white',
                    font='Cambria-Bold',
                    method="label",
                    size=(max_width, None),
                ).set_position('center').set_duration(audio_duration / 2)
                image_clip = CompositeVideoClip([image_clip, text_clip])

            video_clip = image_clip.set_audio(audio_clip)
            video_clips.append(video_clip)

        # Determine the directory of the current file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Create results directory in the same directory as the current file
        parent_dir = os.path.dirname(base_dir)
        parent_results_dir = os.path.join(parent_dir, "results")
        os.makedirs(parent_results_dir, exist_ok=True)

        # Generate a random text component
        random_text = uuid.uuid4().hex[:8]  # Use 8 characters from a UUID

        # Create filename using poem_name and random text
        safe_poem_name = "".join(c for c in poem_name if c.isalnum() or c in [' ', '_', '-']).strip().replace(' ', '_')
        if not safe_poem_name:  # If poem_name is empty or contains only special characters
            safe_poem_name = "poem"

        output_filename = f"{safe_poem_name}_{random_text}.mp4"
        output_file = os.path.abspath(os.path.join(parent_results_dir, output_filename))

        # Write the video file
        final_clip = concatenate_videoclips(video_clips)
        final_clip.write_videofile(output_file, fps=24, codec='libx264', audio_codec='aac')

        # Return the path to the created file
        return output_file

    finally:
        # Clean up resources
        if final_clip:
            final_clip.close()
        for clip in video_clips:
            clip.close()
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        if intro_audio_path and os.path.exists(intro_audio_path):
            os.remove(intro_audio_path)
        os.rmdir(temp_dir)
