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
Generate a structured JSON output for visualizing a segment of a poem. Poem lines, literal explanation, implied intentions and sentiment analysis scores are provided as input. Use these elements to create 8 visualization prompts in the following format:

- 4 literal prompts that faithfully represent the explicit content in the poem
- 4 implied prompts that capture the abstract emotions and subtextual meanings

The sentiment score (-1.0 to 1.0) should influence your color palette and atmosphere:
- Negative scores (-1.0 to -0.3): Use darker, muted colors (deep blues, grays, browns), somber lighting, and heavier visual elements
- Neutral scores (-0.3 to 0.3): Use balanced color schemes, natural lighting, and neutral visual weight
- Positive scores (0.3 to 1.0): Use vibrant, warm colors (yellows, oranges, bright blues), uplifting imagery, and lighter visual elements

For each prompt, provide:
1. A core_element (the central subject/focus)
2. Visual_motifs (3-5 supporting elements that enhance the scene)
3. Style (artistic style that best suits the content - e.g., impressionist, surrealist, photorealistic)
4. A detailed prompt (75-125 words) that combines all elements into a cohesive visual description

For LITERAL prompts:
- Extract concrete nouns, actions, and settings directly from the poem lines
- Maintain the original imagery while enhancing it with sensory details
- Honor the poem's explicit metaphors and similes

For IMPLIED prompts:
- Transform the poem's subtext into abstract, symbolic imagery
- Use visual metaphors that convey the emotional undercurrents
- Create surrealist or expressionist scenes that capture the mood without directly referencing the literal objects

Return only a JSON object with this structure:
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
    "literal4": {
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
    },
    "implied4": {
        "core_element": str,
        "visual_motifs": list[str],
        "style": str,
        "prompt": str
    }
}
"""
