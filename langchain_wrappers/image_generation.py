import os
import replicate
from langchain.tools import BaseTool


class ReplicateImageGenerationTool(BaseTool):
    name: str = "ReplicateImageGenerationTool"
    description: str = (
        "Generates an image using Replicate's 'flux-schnell' model. "
        "Accepts a prompt and returns the image URL. "
        "You can later fetch and display the image."
    )

    def _run(self, prompt: str) -> str:
        # Prepare the input data for the model
        data = {
            "prompt": prompt,
            "output_format": "jpg",
            "aspect_ratio": "16:9"
        }
        # Instantiate the replicate client (uses the REPLICATE_KEY env var)
        client = replicate.Client(api_token=os.getenv("REPLICATE_KEY"))
        # Run the model and assume it returns a list of image URLs
        output = client.run("black-forest-labs/flux-schnell", input=data)
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return output

    async def _arun(self, prompt: str) -> str:
        raise NotImplementedError("Asynchronous run is not supported.")
