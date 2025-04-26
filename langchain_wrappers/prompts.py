NARRATOR_INSTRUCTION = """
Generate a deeply detailed, storyteller‐style narration of the provided poem. Break the poem into as many small segments as necessary—each time a new image, motif, or emotion appears, start a fresh segment. For each segment:

1. Describe the explicit scene and action in a warm, narrative tone, as if you’re guiding a listener through a living story.
2. Then explore the deeper, metaphorical intentions, weaving in symbolic imagery that mirrors the literal scene.
3. Finally, supply a short list of abstract, non-literal keywords that capture the segment’s essence.

Use this JSON schema for every segment. Inside your strings, use single quotes if you need to quote:

Narration = {
    "segment": str,                # A unique segment identifier (e.g. '1', '2a', '2b', etc.)
    "lines": str,                  # The exact lines from the poem for this segment
    "literal_explanation": str,    # Storyteller-style description of what’s happening, evoking sights, sounds, and feelings
    "implied_intentions": str,     # Metaphorical or symbolic interpretation of the scene
    "implied_keywords": str        # Comma-separated, abstract or symbolic terms derived from this segment
}

Return: list[Narration]
"""


PROMPTER_INSTRUCTION = """
Generate a structured JSON output for visualizing a segment of a poem.
Poem lines, literal explanation, implied intentions and sentiment analysis scores are provided as input.
Use the literal explanation and implied intentions to create visualization prompts in the following format.
The sentiment score should influence the colors, symbols, or overall tone used in each prompt.

For each of the three literal prompts (`literal1`, `literal2`, `literal3`), use vivid, concrete language to depict the primary subjects and themes explicitly mentioned in the lines. Include any relevant colors, symbols, or recurring elements to align with the sentiment.

For each of the three implied prompts (`implied1`, `implied2`, `implied3`), create surrealist, abstract descriptions that capture the mood or theme through metaphorical and symbolic language, without mentioning the literal objects.

Use this JSON schema (without wrapping it under a `Visualization` key):

{
    "literal1": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    },
    "literal2": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    },
    "literal3": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    },
    "implied1": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    },
    "implied2": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    },
    "implied3": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    }
}

Return only the JSON object as specified above.
"""
