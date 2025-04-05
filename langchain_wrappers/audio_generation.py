import torch
from TTS.api import TTS

# Determine the device to use (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the TTS model once
tts_model = TTS("tts_models/en/ljspeech/vits").to(device)


def generate_audio(text: str):
    # Generate the audio waveform
    audio = tts_model.tts(text=text)
    return audio

