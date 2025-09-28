from datetime import datetime
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
    scene_number: int
    title: str
    description: str
    duration_seconds: float
    characters_involved: List[str]
    setting: str
    mood: str
    key_dialogue: Optional[str] = None
    visual_notes: Optional[str] = None


class StoryboardData(BaseModel):
    """Model for complete storyboard data."""
    title: str
    total_duration_seconds: float
    scenes: List[StoryboardScene]
    overall_theme: str
    target_audience: str
    created_at: datetime = Field(default_factory=datetime.now)
    source_conversation_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoSegment(BaseModel):
    """Model for individual video segments."""
    segment_number: int
    scene_reference: int  # References StoryboardScene.scene_number
    video_prompt: str
    duration_seconds: float
    style_notes: str
    characters: List[str]
    setting: str
    mood: str


class VideoData(BaseModel):
    """Model for complete video generation data."""
    title: str
    total_duration_seconds: float
    segments: List[VideoSegment]
    overall_style: str
    resolution: str = "1920x1080"
    fps: int = 24
    created_at: datetime = Field(default_factory=datetime.now)
    source_storyboard_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
