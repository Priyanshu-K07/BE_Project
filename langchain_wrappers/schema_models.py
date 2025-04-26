from pydantic import BaseModel, Field
from typing import List


class NarrationSegment(BaseModel):
    segment: str = Field(..., description="The segment number in the sequence of the poem")
    lines: str = Field(..., description="The two-line segment of the poem")
    literal_explanation: str = Field(..., description="Narration of the literal meaning, covering explicit elements and visual motifs")
    implied_intentions: str = Field(..., description="Narration of the implied meaning, revealing deeper or symbolic intentions")
    implied_keywords: str = Field(..., description="List of non-real, abstract, metaphorical representations of lines")


class Narration(BaseModel):
    segments: List[NarrationSegment] = Field(..., description="A list of narration segments for the poem")


class VisualizationSection(BaseModel):
    core_element: str = Field(..., description="The primary subject, object, or theme")
    visual_motifs: List[str] = Field(..., description="Colors, symbols, or recurring elements for visualization")
    style: str = Field(..., description="The artistic style for the image")


class Visualization(BaseModel):
    literal1: VisualizationSection = Field(..., description="First literal visualization details")
    literal2: VisualizationSection = Field(..., description="Second literal visualization details")
    literal3: VisualizationSection = Field(..., description="Second literal visualization details")
    implied1: VisualizationSection = Field(..., description="First implied visualization details")
    implied2: VisualizationSection = Field(..., description="Second implied visualization details")
    implied3: VisualizationSection = Field(..., description="Second implied visualization details")

