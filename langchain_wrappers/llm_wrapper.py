import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


class GeminiLLM:
    """
    A wrapper for Google's Gemini LLM using ChatGoogleGenerativeAI.
    """

    def __init__(self, system_instruction: str, model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
        """
        Initializes the GeminiLLM wrapper.

        Parameters:
          system_instruction (str): Instructions for the model (narrator or prompter).
          model_name (str): The Google model name (e.g., "gemini-1.5-pro").
          temperature (float): Temperature setting for generation.
        """
        self.system_instruction = system_instruction
        self.model_name = model_name
        self.temperature = temperature
        # The ChatGoogleGenerativeAI instance automatically reads GOOGLE_API_KEY from environment if not passed.
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key,
        )

    def generate_content(self, prompt: str) -> str:
        """
        Generates content by combining the system instruction with the prompt.

        Parameters:
          prompt (str): The user-provided prompt or input.

        Returns:
          str: The generated response content.
        """
        # Build the message sequence with the system instruction first, then the prompt.
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "human", "content": prompt}
        ]
        response = self.llm.invoke(messages)
        return response.content

    async def agenerate_content(self, prompt: str) -> str:
        """
        Asynchronous version to generate content.

        Parameters:
          prompt (str): The user prompt.

        Returns:
          str: The generated response content.
        """
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "human", "content": prompt}
        ]
        response = await self.llm.ainvoke(messages)
        return response.content



