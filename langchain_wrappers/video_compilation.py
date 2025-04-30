import os
import tempfile
import uuid
import random
import time
import shutil
from typing import List

import moviepy.config as mpy_config
import numpy as np
import scipy.io.wavfile as wav
import torch
from PIL import Image
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips
)
from moviepy.video.fx.resize import resize
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout

from langchain_wrappers.audio_generation import generate_audio

mpy_config.change_settings({
    "IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
})


def create_video_from_images_and_audio(
        images: List,
        audio_tensors: List,
        sample_rate: int = 22050,  # Keep your original sample rate
        poem_lines: List[str] = None,
        poet: str = None,
        poem_name: str = "",
        images_per_audio: int = 4
) -> str:
    """
    Create a cinematic video from PIL images and audio tensors with an introductory title screen.
    Applies Ken Burns zoom, fade transitions, and alternating poem-line overlays.
    Returns the output file path of the generated MP4.
    """
    # Validate image count
    expected_images = len(audio_tensors) * images_per_audio
    if len(images) != expected_images:
        raise ValueError(
            f"Expected {expected_images} images for {len(audio_tensors)} audio clips ("
            f"{images_per_audio} images each), got {len(images)}"
        )

    # Create a unique temporary directory with a random name to avoid conflicts
    temp_dir = tempfile.mkdtemp(prefix=f"poem_video_{uuid.uuid4().hex[:6]}_")

    video_clips = []
    audio_clips = []  # Keep track of audio clip objects
    audio_paths = []
    intro_audio_path = None

    # Track all temporary files for cleanup
    temp_files = []

    try:
        # --------------------
        # Introductory title screen
        # --------------------
        intro_text = f"{poem_name} by {poet}" if poet else poem_name
        intro_audio = generate_audio(intro_text)
        intro_audio_tensor = torch.tensor(intro_audio, dtype=torch.float32)

        # Use unique filenames for all temporary files
        intro_audio_path = os.path.join(temp_dir, f"intro_audio_{uuid.uuid4().hex[:8]}.wav")
        temp_files.append(intro_audio_path)

        intro_array = (intro_audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        wav.write(intro_audio_path, sample_rate, intro_array)

        # Wait briefly to ensure file is completely written before opening
        time.sleep(0.1)

        intro_audio_clip = AudioFileClip(intro_audio_path)
        audio_clips.append(intro_audio_clip)
        intro_duration = intro_audio_clip.duration

        # Title with fade-in/out
        title_clip = TextClip(
            poem_name,
            fontsize=80,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center',
            size=(1100, None)
        ).set_position(("center", 100)).set_duration(intro_duration)

        # Apply fades to title
        title_clip = title_clip.fx(fadein, 1.0).fx(fadeout, 1.0)

        # Poet credit with fade-in/out
        poet_clip = TextClip(
            f"by {poet}" if poet else "",
            fontsize=40,
            color='white',
            font='Cambria-Bold',
            method="label",
            align='center'
        ).set_position(("center", 100 + title_clip.h + 20)).set_duration(intro_duration)

        # Apply fades to poet credit
        poet_clip = poet_clip.fx(fadein, 1.0).fx(fadeout, 1.0)

        bg = ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8)).set_duration(intro_duration)
        title_screen = CompositeVideoClip([bg, title_clip, poet_clip]).set_audio(intro_audio_clip)
        video_clips.append(title_screen)

        # --------------------
        # Convert audio tensors to WAV files with unique names
        # --------------------
        for i, audio_tensor in enumerate(audio_tensors):
            unique_id = uuid.uuid4().hex[:8]
            audio_path = os.path.join(temp_dir, f"audio_{i}_{unique_id}.wav")
            temp_files.append(audio_path)

            # Save with proper scaling for 16-bit wav
            arr = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
            wav.write(audio_path, sample_rate, arr)
            audio_paths.append(audio_path)

            # Wait briefly to ensure file is completely written
            time.sleep(0.1)

        # --------------------
        # Create video segments
        # --------------------
        render_text = True
        poem_line_idx = 0

        for idx, audio_path in enumerate(audio_paths):
            try:
                audio_clip = AudioFileClip(audio_path)
                audio_clips.append(audio_clip)
                audio_duration = audio_clip.duration

                # Calculate exact duration needed for each image to match audio perfectly
                sub_duration = audio_duration / images_per_audio

                start = idx * images_per_audio
                batch = images[start:start + images_per_audio]
                subclips = []

                for j, img in enumerate(batch):
                    img = img.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
                    arr = np.array(img)

                    # Create image clip with exact duration
                    clip = ImageClip(arr).set_duration(sub_duration).set_fps(24)

                    # Apply Ken Burns effect (subtle zoom) with error handling
                    try:
                        zoom_factor = random.uniform(0.01, 0.02)  # Subtle zoom
                        clip = clip.fx(resize, lambda t: 1 + zoom_factor * t)
                    except Exception as e:
                        print(f"Warning: Ken Burns effect failed, using static image: {e}")
                        # Continue with static image if Ken Burns fails

                    # Calculate fade duration based on clip length
                    # Make sure fades don't eat up too much of the clip time
                    fade_dur = min(0.3, sub_duration / 6)  # Max 0.3s or 1/6 of clip duration

                    # Apply fades
                    clip = clip.fx(fadein, fade_dur).fx(fadeout, fade_dur)

                    # Add text overlay if needed
                    if (poem_lines and render_text and j == 0 and poem_line_idx < len(poem_lines)):
                        text = poem_lines[poem_line_idx].upper()
                        poem_line_idx += 1

                        max_w = int(arr.shape[1] * 0.9)
                        txt_clip = TextClip(
                            text,
                            fontsize=60,
                            color='white',
                            font='Cambria-Bold',
                            method="label",
                            size=(max_w, None)
                        ).set_position('center').set_duration(sub_duration)

                        # Apply fades to text (matching the image fade timing)
                        txt_clip = txt_clip.fx(fadein, fade_dur).fx(fadeout, fade_dur)

                        clip = CompositeVideoClip([clip, txt_clip])

                    subclips.append(clip)

                # Toggle text rendering for next segment
                render_text = not render_text

                # Concatenate clips precisely for this audio segment
                segment_clip = concatenate_videoclips(subclips, method="compose")

                # Set audio AFTER concatenation to ensure sync
                segment_clip = segment_clip.set_audio(audio_clip)
                video_clips.append(segment_clip)

            except Exception as e:
                print(f"Error processing segment {idx}: {e}")
                # Continue with other segments if possible
                continue

        # --------------------
        # Final assembly and output
        # --------------------
        base = os.path.dirname(os.path.abspath(__file__))
        results = os.path.join(os.path.dirname(base), "results")
        os.makedirs(results, exist_ok=True)

        rand = uuid.uuid4().hex[:8]
        safe_name = "".join(c for c in poem_name if c.isalnum() or c in [' ', '_', '-']).strip().replace(' ',
                                                                                                         '_') or "poem"
        out_name = f"{safe_name}_{rand}.mp4"
        out_path = os.path.join(results, out_name)

        # Create final video with all segments
        final = concatenate_videoclips(video_clips, method="compose")

        # Create a unique temporary directory for output processing
        temp_output_dir = tempfile.mkdtemp(prefix=f"poem_output_{uuid.uuid4().hex[:6]}_")
        temp_audio_path = os.path.join(temp_output_dir, f"temp_audio_{uuid.uuid4().hex[:8]}.m4a")
        temp_files.append(temp_audio_path)

        try:
            # Export with high quality settings and unique temp paths
            final.write_videofile(
                out_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                audio_bitrate='256k',  # Higher audio bitrate for better quality
                preset='medium',  # Better quality encoding
                threads=4,  # Multi-threading for efficiency
                temp_audiofile=temp_audio_path,
                remove_temp=False  # We'll handle cleanup ourselves
            )

            return out_path

        except Exception as e:
            print(f"Error writing final video: {e}")
            # If writing fails, try with safer settings
            try:
                print("Trying with safer settings...")
                final.write_videofile(
                    out_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    threads=1,  # Single thread
                    temp_audiofile=temp_audio_path,
                    remove_temp=False
                )
                return out_path
            except Exception as e2:
                print(f"Final attempt failed: {e2}")
                raise

    except Exception as main_error:
        print(f"Error in video creation: {main_error}")
        raise

    finally:
        # Close all clips first to release resources
        print("Cleaning up resources...")

        # Close all audio clips
        for clip in audio_clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except:
                pass

        # Close all video clips
        for clip in video_clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except:
                pass

        # Wait a moment to ensure all resources are released
        time.sleep(0.5)

        # Clean up temporary files
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove temp file {file_path}: {e}")

        # Clean up temporary directories
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Failed to remove temp directory {temp_dir}: {e}")

        try:
            if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir, ignore_errors=True)
        except Exception as e:
            print(f"Failed to remove temp output directory {temp_output_dir}: {e}")