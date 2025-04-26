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
        poem_name: str = "",
        images_per_audio: int = 3
) -> str:
    """
    Create a video from lists of PIL images and audio tensors with an introductory title screen.
    Displays `images_per_audio` images for each audio clip by splitting the audio duration evenly.
    Returns the output video file path.
    """
    # Validate image count
    expected_images = len(audio_tensors) * images_per_audio
    if len(images) != expected_images:
        raise ValueError(
            f"Expected {expected_images} images for {len(audio_tensors)} audio clips ("
            f"{images_per_audio} images each), got {len(images)}"
        )

    # Prepare temporary storage
    temp_dir = tempfile.mkdtemp()
    video_clips = []
    audio_paths = []
    intro_audio_path = None

    try:
        # --------------------
        # Introductory title screen
        # --------------------
        intro_text = f"{poem_name} by {poet}" if poet else poem_name
        intro_audio = generate_audio(intro_text)
        intro_audio_tensor = torch.tensor(intro_audio, dtype=torch.float32)
        intro_audio_path = os.path.join(temp_dir, "intro_audio.wav")
        intro_array = (intro_audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        wav.write(intro_audio_path, sample_rate, intro_array)

        intro_audio_clip = AudioFileClip(intro_audio_path)
        intro_duration = intro_audio_clip.duration

        title_clip = TextClip(
            poem_name,
            fontsize=80,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center',
            size=(1100, None)
        ).set_position(("center", 100)).set_duration(intro_duration)

        poet_clip = TextClip(
            f"by {poet}" if poet else "",
            fontsize=40,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center'
        ).set_position(("center", 100 + title_clip.h + 20)).set_duration(intro_duration)

        bg = ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8)).set_duration(intro_duration)
        title_screen = CompositeVideoClip([bg, title_clip, poet_clip]).set_audio(intro_audio_clip)
        video_clips.append(title_screen)

        # --------------------
        # Convert audio tensors to WAV
        # --------------------
        for i, audio_tensor in enumerate(audio_tensors):
            path = os.path.join(temp_dir, f"audio_{i}.wav")
            arr = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
            wav.write(path, sample_rate, arr)
            audio_paths.append(path)

        # --------------------
        # Create video segments
        # --------------------

        render_text = True
        poem_line_idx = 0

        for idx, audio_path in enumerate(audio_paths):
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            sub_duration = duration / images_per_audio

            start = idx * images_per_audio
            batch = images[start:start + images_per_audio]
            subclips = []

            for j, img in enumerate(batch):
                img = img.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
                arr = np.array(img)
                clip = ImageClip(arr).set_duration(sub_duration).set_fps(24)

                # only on the first frame of the batch, when flag is True,
                # and we still have lines left
                if (
                        poem_lines
                        and render_text
                        and j == 0
                        and poem_line_idx < len(poem_lines)
                ):
                    text = poem_lines[poem_line_idx].upper()
                    poem_line_idx += 1  # consume exactly one line here

                    max_w = int(arr.shape[1] * 0.9)
                    txt_clip = TextClip(
                        text,
                        fontsize=60,
                        color='white',
                        font='Cambria-Bold',
                        method="label",
                        size=(max_w, None)
                    ).set_position('center').set_duration(sub_duration)
                    clip = CompositeVideoClip([clip, txt_clip])

                subclips.append(clip)

            # flip for next batch
            render_text = not render_text

            segment_clip = concatenate_videoclips(subclips).set_audio(audio_clip)
            video_clips.append(segment_clip)

        # --------------------
        # Output file setup
        # --------------------
        base = os.path.dirname(os.path.abspath(__file__))
        results = os.path.join(os.path.dirname(base), "results")
        os.makedirs(results, exist_ok=True)

        rand = uuid.uuid4().hex[:8]
        safe_name = "".join(c for c in poem_name if c.isalnum() or c in [' ', '_', '-']).strip().replace(' ', '_') or "poem"
        out_name = f"{safe_name}_{rand}.mp4"
        out_path = os.path.join(results, out_name)

        final = concatenate_videoclips(video_clips)
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')

        return out_path

    finally:
        # Cleanup
        for clip in video_clips:
            clip.close()
        if intro_audio_path and os.path.exists(intro_audio_path):
            os.remove(intro_audio_path)
        for p in audio_paths:
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(temp_dir)
