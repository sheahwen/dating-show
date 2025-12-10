from datetime import datetime
from email.policy import default
from tkinter import N
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    DIRECTOR = "director"
    PARTICIPANT = "participant"


class ConversationMessage(BaseModel):
    """Model for individual conversation messages."""
    content: str
    source: str
    message_type: MessageType
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ParticipantInfo(BaseModel):
    """Model for participant information."""
    name: str
    age: int
    personality_traits: List[str]
    system_message: str


class DirectorInfo(BaseModel):
    """Model for director information."""
    name: str
    system_message: str


class ConversationData(BaseModel):
    """Model for complete conversation data."""
    participants: List[ParticipantInfo]
    director: DirectorInfo
    messages: List[ConversationMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    max_turns: int = 5
    total_turns: int = 0


class StoryboardScene(BaseModel):
    """Model for individual storyboard scenes."""
    scene_number: int = Field(description="Sequential number of this scene within the episode")
    title: str = Field(description="The title of the scene")
    description: str = Field(description="Detailed description of what happens in the scene")
    duration_seconds: float = Field(description="The duration of the scene in seconds")
    characters_involved: List[str] = Field(description="List of character names appearing in the scene")
    setting: str = Field(description="Physical location or environment")
    mood: str = Field(description="Emotional tone and atmosphere (e.g., 'romantic', 'tense', 'playful')")
    camera_shot: Optional[str] = Field(default=None, description="Suggested camera shot type (e.g., 'close-up', 'wide shot', 'medium shot')")
    key_dialogue: Optional[str] = Field(default=None, description="Important dialogue if any")
    visual_notes: Optional[str] = Field(default=None, description="Camera and visual notes")


class StoryboardData(BaseModel):
    """Model for complete storyboard data."""
    title: str = Field(description="The title of episode")
    total_duration_seconds: float = Field(description="The total duration of the episode in seconds")
    scenes: List[StoryboardScene] = Field(description="The list of scenes in the episode")
    overall_theme: str = Field(description="The overall theme of the episode")
    created_at: datetime = Field(default_factory=datetime.now, description="The timestamp of the episode")
    source_conversation_file: str = Field(description="The conversation file that the storyboard is based on")

class StoryboardMetadata(BaseModel):
    """Model for metadata of a storyboard."""
    pipeline_stage: int = Field(description="The stage of the pipeline that generated the storyboard")
    generation_method: str = Field(description="The method used to generate the storyboard")
    model_used: str = Field(description="The model used to generate the storyboard")
    participants: List[str] = Field(description="The list of participants in the storyboard")
    created_at: datetime = Field(default_factory=datetime.now, description="The timestamp of the storyboard")

class StoryboardWithMetadataData(StoryboardData):
    """Model for complete storyboard data with metadata."""
    metadata: StoryboardMetadata = Field(description="The metadata of the storyboard")

class VideoSegment(BaseModel):
    """Model for individual video segments used for AI video generation."""
    segment_number: int = Field(description="Sequential number of this video segment within the episode")
    scene_reference: int = Field(description="References the scene_number from the corresponding StoryboardScene")
    video_prompt: str = Field(description="Video generation prompt for AI video tools")
    duration_seconds: float = Field(description="Duration of this video segment in seconds")
    camera_shot: str = Field(description="Camera shot type and angle (e.g., 'close-up', 'wide shot', 'medium shot', 'over-the-shoulder', 'establishing shot')")
    style_notes: str = Field(description="Visual style and cinematography notes")
    characters: List[str] = Field(description="List of character names appearing")
    setting: str = Field(description="Physical location or environment")
    mood: str = Field(description="Emotional tone and atmosphere (e.g., 'romantic', 'tense', 'playful')")


class VideoData(BaseModel):
    """Model for complete video generation data containing all segments and metadata."""
    title: str = Field(description="Title of the video episode")
    total_duration_seconds: float = Field(description="Total duration of the complete video in seconds")
    segments: List[VideoSegment] = Field(description="List of video segments that make up the complete episode")
    overall_style: str = Field(description="Overall visual style and aesthetic for the entire video")
    resolution: str = Field(default="1920x1080", description="Video resolution in WxH format")
    fps: int = Field(default=24, description="Frames per second for video playback")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when this video data was created")
    source_storyboard_file: str = Field(description="Path to the storyboard file this video data was generated from")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the video generation process")
