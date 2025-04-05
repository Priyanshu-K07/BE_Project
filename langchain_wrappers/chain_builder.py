import asyncio
import io
import requests
import torch
from PIL import Image
from TTS.api import TTS
from langchain.chains.base import Chain
from langchain.output_parsers import PydanticOutputParser
from typing import Dict, Any, List
import json
from langchain_wrappers.image_generation import ReplicateImageGenerationTool
from langchain_wrappers.llm_wrapper import GeminiLLM
from langchain_wrappers.video_compilation import create_video_from_images_and_audio
import nltk
from nltk.data import find
from nltk.sentiment import SentimentIntensityAnalyzer


def download_nltk_resource(resource_path: str, download_name: str):
    try:
        find(resource_path)
    except LookupError:
        nltk.download(download_name)


# Check and download only if missing
download_nltk_resource('sentiment/vader_lexicon', 'vader_lexicon')
download_nltk_resource('tokenizers/punkt', 'punkt')


# === Utility Functions ===
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


def generate_audio(text: str) -> Any:
    """
    Generate audio from text using Coqui TTS.
    Uses VITS model for natural synthesis.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/en/ljspeech/vits").to(device)
    return tts.tts(text)


# Helper function: Convert an image URL to a PIL Image.
def url_to_pil(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to fetch image from URL: {url} with status {response.status_code}")


# Helper function to format the prompt dictionary into a text prompt.
def format_prompt(prompt_section: Dict[str, Any]) -> str:
    """
    Converts a prompt section (with keys: 'core_element', 'visual_motifs', 'style')
    into a formatted string.
    """
    core = prompt_section.get("core_element", "")
    motifs = prompt_section.get("visual_motifs", [])
    style = prompt_section.get("style", "")
    motifs_str = ", ".join(motifs) if motifs else ""
    return f"{core}. Visual motifs: {motifs_str}. Style: {style}."


# === CHAIN 1: Poem Analysis (Narration + Sentiment) ===
class PoemAnalysisChain(Chain):
    """
    1. Calls Gemini LLM for structured narration of a poem.
    2. Preprocesses and parses output using the `Narration` schema.
    3. Performs sentiment analysis on the poem.
    """
    llm: GeminiLLM
    output_parser: PydanticOutputParser

    @property
    def input_keys(self) -> List[str]:
        return ["poem_text"]

    @property
    def output_keys(self) -> List[str]:
        return ["narration", "sentiment"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        poem_text = inputs["poem_text"]
        print(f"{poem_text=}")

        # Generate structured narration from the LLM
        response_text = self.llm.generate_content(poem_text)
        print(f"{response_text=}")

        # Preprocess the response text:
        # 1. Remove markdown code fences if present.
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()

        # 2. Attempt to load the cleaned text as JSON.
        try:
            parsed_json = json.loads(cleaned_text)
        except Exception as e:
            raise ValueError(f"Failed to load JSON from LLM output: {e}")

        # 3. If the parsed JSON is a list, wrap it into a dict with key "segments".
        if isinstance(parsed_json, list):
            parsed_json = {"segments": parsed_json}

        # Now convert the dict back to a JSON string for the output parser.
        json_for_parser = json.dumps(parsed_json)

        # Parse structured output using the output parser (which expects the Narration schema)
        try:
            narration_parsed = self.output_parser.parse(json_for_parser)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")

        # Perform sentiment analysis on the original poem text
        sentiment = analyze_sentiment(poem_text)

        return {
            "narration": narration_parsed.dict(),
            "sentiment": sentiment
        }


# --- IMAGE PROMPT CHAIN ---
class ImagePromptChain(Chain):
    """
    For each narration segment:
      1. Uses the Gemini LLM (with an output parser) to generate visualization prompts.
         The parsed output is expected to be a dictionary with keys "literal" and "implied".
      2. For each section, it formats the prompt into a text string.
      3. It calls the imported ReplicateImageGenerationTool to generate an image URL.
      4. It converts the image URL into a PIL Image.
      5. Returns a list (one entry per segment) with keys "literal" and "implied" containing the images.
    """
    llm: GeminiLLM  # Your GeminiLLM instance
    output_parser: PydanticOutputParser  # Your PydanticOutputParser instance configured with the Visualization schema

    @property
    def input_keys(self) -> List[str]:
        return ["narration_segments"]

    @property
    def output_keys(self) -> List[str]:
        return ["images"]

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        narration_segments = inputs["narration_segments"]
        images = []
        # Instantiate the Replicate tool (imported from your module)
        replicate_tool = ReplicateImageGenerationTool()

        for segment in narration_segments:
            print("Image generation")
            # Build a prompt message (here using segment details; adjust as needed)
            # For demonstration, we're simply passing the segment's lines to generate a visualization prompt.
            msg = f"Generate visualization prompt for: {segment['lines']}"
            response_text = self.llm.generate_content(msg)
            try:
                parsed_prompt = self.output_parser.parse(response_text)
            except Exception as e:
                raise ValueError(f"Failed to parse image prompt: {e}")
            # Expecting the parsed prompt to be a dict with keys 'literal' and 'implied'
            prompt_dict = parsed_prompt.dict()
            literal_prompt_str = format_prompt(prompt_dict.get("literal", {}))
            implied_prompt_str = format_prompt(prompt_dict.get("implied", {}))

            # Call the replicate tool to generate image URLs.
            literal_image_url = replicate_tool.run(literal_prompt_str)
            implied_image_url = replicate_tool.run(implied_prompt_str)

            # Convert image URLs to PIL Images.
            literal_image = url_to_pil(literal_image_url)
            implied_image = url_to_pil(implied_image_url)

            # images.append({
            #     "literal": literal_image,
            #     "implied": implied_image
            # })
            images.append(literal_image)
            images.append(implied_image)

        return {"images": images}

    async def _arun(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        narration_segments = inputs["narration_segments"]
        images = []
        replicate_tool = ReplicateImageGenerationTool()

        for segment in narration_segments:
            print("Async Image generation")
            msg = f"Generate visualization prompt for: {segment['lines']}"
            response_text = await asyncio.to_thread(self.llm.generate_content, msg)
            try:
                parsed_prompt = self.output_parser.parse(response_text)
            except Exception as e:
                raise ValueError(f"Failed to parse image prompt: {e}")
            prompt_dict = parsed_prompt.dict()
            literal_prompt_str = format_prompt(prompt_dict.get("literal", {}))
            implied_prompt_str = format_prompt(prompt_dict.get("implied", {}))

            literal_image_url = await asyncio.to_thread(replicate_tool.run, literal_prompt_str)
            implied_image_url = await asyncio.to_thread(replicate_tool.run, implied_prompt_str)

            literal_image = await asyncio.to_thread(url_to_pil, literal_image_url)
            implied_image = await asyncio.to_thread(url_to_pil, implied_image_url)

            # images.append({
            #     "literal": literal_image,
            #     "implied": implied_image
            # })
            images.append(literal_image)
            images.append(implied_image)
        return {"images": images}


class AudioGenerationChain(Chain):
    """
    1. Accepts text inputs.
    2. Generates audio using TTS.
    3. Converts numpy arrays to tensors and returns generated audio list.
    """

    @property
    def input_keys(self) -> List[str]:
        return ["audio_texts"]

    @property
    def output_keys(self) -> List[str]:
        return ["audios"]

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        segments = inputs["audio_texts"]["segments"]
        audios = []
        for segment in segments:
            print("Audio Generation")
            # Generate audio for combined lines and literal explanation
            audio_np1 = generate_audio(segment['lines'] + '...' + segment['literal_explanation'])
            audio_tensor1 = torch.tensor(audio_np1, dtype=torch.float32)
            audios.append(audio_tensor1)

            # Generate audio for implied intentions
            audio_np2 = generate_audio(segment['implied_intentions'])
            audio_tensor2 = torch.tensor(audio_np2, dtype=torch.float32)
            audios.append(audio_tensor2)
        return {"audios": audios}

    async def _arun(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        segments = inputs["audio_texts"]["segments"]
        audios = []
        for segment in segments:
            print("Async Audio Generation")
            audio_np1 = await asyncio.to_thread(generate_audio, segment['lines'] + '...' + segment['literal_explanation'])
            audio_tensor1 = torch.tensor(audio_np1, dtype=torch.float32)
            audios.append(audio_tensor1)

            audio_np2 = await asyncio.to_thread(generate_audio, segment['implied_intentions'])
            audio_tensor2 = torch.tensor(audio_np2, dtype=torch.float32)
            audios.append(audio_tensor2)

        return {"audios": audios}


# === CHAIN 4: Video Compilation ===
class VideoCompilationChain(Chain):
    """
    1. Accepts images and audio tensors.
    2. Compiles them into a video.
    """
    @property
    def input_keys(self) -> List[str]:
        return ["images", "audios", "poet", "poem_name", "poem_lines"]

    @property
    def output_keys(self) -> List[str]:
        return ["video_path"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        video_path = create_video_from_images_and_audio(
            images=inputs["images"],
            audio_tensors=inputs["audios"],
            poet=inputs.get("poet", ""),
            poem_name=inputs.get("poem_name", ""),
            poem_lines=inputs.get("poem_lines", None)
        )
        return {"video_path": video_path}

    async def _arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Asynchronous run is not supported.")
