NARRATOR_INSTRUCTION = """
Generate a detailed storyteller-style narration of the provided poem. For each two-line segment, cover the literal meaning first, followed by the implied intentions.
The implied intentions should contain metaphorical representations of the literal meaning.
Keep the narration in a natural style and ensure consistency between segments for a smooth narrative flow.

Use this JSON schema to structure the narration. Avoid using double quotes inside strings; use single quotes instead for any internal quotations.

Narration = {
    "segment": str,                # The segment number in the sequence of the poem
    "lines": str,                  # The two-line segment of the poem
    "literal_explanation": str,     # Narration of the literal meaning, covering explicit elements and visual motifs
    "implied_intentions": str,      # Narration of the implied meaning, revealing deeper or symbolic intentions
    "implied_keywords": str,       # List of non-real, abstract, metaphorical representations of lines
}

Return: list[Narration]
"""

PROMPTER_INSTRUCTION = """
Generate a structured JSON output for visualizing a two-line segment of a poem.
Poem lines and sentiment analysis score are provided as input.
The sentiment score should influence the colors, symbols, or overall tone used in the output.

For the literal section, use vivid, concrete language to depict the primary subjects and themes explicitly mentioned in the lines. Include any relevant colors, symbols, or recurring elements to align with the sentiment.

For the implied section, create a prompt that abstractly represents the mood or theme of the lines through metaphorical and symbolic language. Use surrealist, abstract descriptions that connect emotionally or conceptually without mentioning the literal objects.
I want the implied prompt to be in alignment with the implied intentions provided.

Use this JSON schema:

Visualization = {
    'literal': {
        'core_element': str,        # The primary subject, object, or theme explicitly mentioned in the line
        'visual_motifs': list[str], # Colors, symbols, or recurring elements for literal representation, influenced by sentiment
        'style': str                # The artistic style for the literal image
    },
    'implied': {
        'core_element': str,        # The primary subject, object, or theme implied by the line
        'visual_motifs': list[str], # Colors, symbols, or recurring elements for implied representation, influenced by sentiment
        'style': str                # The artistic style for the implied image (always Surrealistic abstract painting)
    }
}

Return: Visualization
"""