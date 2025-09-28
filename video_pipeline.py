import asyncio
import json
import os
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError
from autogen_core.models import LLMMessage, UserMessage, AssistantMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import extract_content_without_thoughts


class SceneSegment(BaseModel):
    """Represents a single scene segment with timing and visual details."""
    speaker: str = Field(description="Name of the person speaking")
    dialogue: str = Field(description="What the speaker says")
    duration_seconds: float = Field(gt=0, description="Duration in seconds")
    scene_description: str = Field(description="Detailed visual description")
    camera_angle: str = Field(description="Type of camera shot")
    mood: str = Field(description="Emotional tone of the scene")
    visual_elements: List[str] = Field(description="Specific visual details")
    character_emotions: Dict[str, str] = Field(description="Character emotional states")
    background_setting: str = Field(description="Scene location/setting")
    lighting: str = Field(description="Lighting style and mood")


class Storyboard(BaseModel):
    """Complete storyboard for the dating show episode."""
    episode_title: str = Field(description="Title of the episode")
    total_duration: float = Field(gt=0, description="Total duration in seconds")
    overall_mood: str = Field(description="Overall emotional tone")
    character_descriptions: Dict[str, str] = Field(description="Physical descriptions of characters")
    scene_segments: List[SceneSegment] = Field(description="List of scene segments")
    transitions: List[str] = Field(description="Types of transitions between scenes")
    music_suggestions: List[str] = Field(description="Suggested music tracks")
    
    def save_to_file(self, filepath: str) -> None:
        """Save storyboard to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)


class StoryboardGenerator:
    """Stage 2: Converts conversation into detailed storyboard using LLM."""
    
    def __init__(self, model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]):
        self.model_client = model_client
        self.max_retries = 3
        
    async def generate_storyboard(self, messages: List[LLMMessage], 
                                participant_details: List[Dict]) -> Storyboard:
        """Generate detailed storyboard from conversation."""
        
        # Extract clean conversation
        conversation_text = self._extract_conversation(messages)
        print(f"Conversation text: {conversation_text}")
        
        # Get character information
        character_info = self._format_character_info(participant_details)
        
        # Generate storyboard using LLM with retries
        for attempt in range(self.max_retries):
            try:
                print(f"🎬 Attempt {attempt + 1}/{self.max_retries} to generate storyboard...")
                
                storyboard_prompt = self._create_storyboard_prompt(conversation_text, character_info)
                response = await self._call_storyboard_llm(storyboard_prompt)
                
                # Parse and validate with Pydantic
                storyboard = self._parse_and_validate_response(response)
                print(f"✅ Successfully generated valid storyboard!")
                return storyboard
                
            except ValidationError as e:
                print(f"❌ Validation error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    print(f"🔄 All attempts failed, using fallback storyboard")
                    return self._create_fallback_storyboard_object(messages)
                else:
                    print(f"🔄 Retrying with more specific instructions...")
                    
            except Exception as e:
                print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return self._create_fallback_storyboard_object(messages)
        
        return self._create_fallback_storyboard_object(messages)
    
    def _extract_conversation(self, messages: List[LLMMessage]) -> str:
        """Extract clean conversation text."""
        conversation = []
        for msg in messages[1:]:  # Skip initial producer message
            speaker = msg.source
            content = extract_content_without_thoughts(msg.content)
            conversation.append(f"{speaker}: {content}")
        
        return "\n\n".join(conversation)
    
    def _format_character_info(self, participant_details: List[Dict]) -> str:
        """Format character information for the LLM."""
        char_info = []
        for participant in participant_details:
            traits = ", ".join(participant.get('personality_traits', []))
            char_info.append(
                f"- {participant['name']} (age {participant['age']}): {traits}"
            )
        return "\n".join(char_info)
    
    def _create_storyboard_prompt(self, conversation: str, characters: str) -> str:
        """Create comprehensive storyboard prompt for LLM."""
        
        # Generate JSON schema from Pydantic model
        schema = Storyboard.model_json_schema()
        
        return f"""You are a professional TV director creating a storyboard for a reality dating show episode.

CHARACTER INFORMATION:
{characters}

CONVERSATION TO STORYBOARD:
{conversation}

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON that matches the exact schema below
2. Do NOT include any text before or after the JSON
3. Do NOT use markdown code blocks or backticks
4. Ensure all required fields are present
5. Use realistic durations (5-30 seconds per segment)

JSON SCHEMA TO FOLLOW:
{json.dumps(schema, indent=2)}

EXAMPLE RESPONSE FORMAT:
{{
  "episode_title": "First Connections",
  "total_duration": 180,
  "overall_mood": "romantic_playful",
  "character_descriptions": {{
    "Alex": "Confident young man with bright eyes and charming smile",
    "Sarah": "Artistic woman with gentle demeanor and thoughtful expressions"
  }},
  "scene_segments": [
    {{
      "speaker": "Alex",
      "dialogue": "I'm Alex, 28, from New York City.",
      "duration_seconds": 12.0,
      "scene_description": "Close-up of Alex introducing himself with confidence",
      "camera_angle": "close_up",
      "mood": "confident_welcoming",
      "visual_elements": ["confident smile", "direct eye contact"],
      "character_emotions": {{"Alex": "excited_confident"}},
      "background_setting": "villa_entrance",
      "lighting": "natural_warm"
    }}
  ],
  "transitions": ["fade_in", "cut"],
  "music_suggestions": ["romantic_piano"]
}}

Now create the storyboard JSON:"""

    async def _call_storyboard_llm(self, prompt: str) -> str:
        """Call LLM to generate storyboard."""
        try:
            # Enhanced system message for JSON output
            system_message = UserMessage(
                content="""You are a professional TV director and storyboard artist. 
                
CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no extra text.
Your entire response must be parseable JSON that matches the provided schema exactly.""",
                source="system"
            )
            user_message = UserMessage(content=prompt, source="storyboard_generator")
            
            # Call the model client
            response = await self.model_client.create([system_message, user_message])
            
            # Extract content based on response type
            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
            elif hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
            else:
                raise AttributeError(f"Unable to extract content from response: {type(response)}")
            
            return content.strip()
            
        except Exception as e:
            print(f"Error calling storyboard LLM: {e}")
            raise
    
    def _parse_and_validate_response(self, response: str) -> Storyboard:
        """Parse and validate LLM response using Pydantic."""
        try:
            print(f"🔍 Raw response length: {len(response)} characters")
            print(f"🔍 Raw response preview: {response[:200]}...")
            
            # Clean the response
            cleaned_response = self._clean_json_string(response)
            
            # Parse JSON
            data = json.loads(cleaned_response)
            
            # Validate with Pydantic
            storyboard = Storyboard(**data)
            
            print(f"✅ Successfully validated storyboard with {len(storyboard.scene_segments)} segments")
            return storyboard
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"🔍 Problematic content around error:")
            self._debug_json_error(cleaned_response, e)
            raise
            
        except ValidationError as e:
            print(f"❌ Pydantic validation error:")
            for error in e.errors():
                print(f"  - {error['loc']}: {error['msg']}")
            raise
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove markdown code blocks
        json_str = json_str.replace('```json', '').replace('```', '')
        
        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]
        
        # Remove any text after the last }
        end_idx = json_str.rfind('}')
        if end_idx != -1:
            json_str = json_str[:end_idx + 1]
        
        # Fix common JSON issues
        json_str = json_str.replace('\n', ' ')  # Remove newlines within strings
        json_str = json_str.replace('\t', ' ')  # Remove tabs
        
        # Fix trailing commas (basic approach)
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str.strip()
    
    def _debug_json_error(self, json_str: str, error: json.JSONDecodeError) -> None:
        """Debug JSON parsing errors."""
        lines = json_str.split('\n')
        start_line = max(0, error.lineno - 3)
        end_line = min(len(lines), error.lineno + 2)
        
        print(f"Context around line {error.lineno}, column {error.colno}:")
        for i in range(start_line, end_line):
            marker = ">>> " if i == error.lineno - 1 else "    "
            if i < len(lines):
                print(f"{marker}Line {i+1}: {lines[i]}")
        
        # Save for debugging
        with open("debug_invalid_json.txt", "w") as f:
            f.write(json_str)
        print(f"💾 Saved problematic JSON to debug_invalid_json.txt")
    
    def _create_fallback_storyboard_object(self, messages: List[LLMMessage]) -> Storyboard:
        """Create a basic fallback storyboard object using Pydantic."""
        segments = []
        for i, msg in enumerate(messages[1:]):  # Skip producer message
            segments.append(SceneSegment(
                speaker=msg.source,
                dialogue=extract_content_without_thoughts(msg.content)[:100] + "...",
                duration_seconds=15.0,
                scene_description=f"{msg.source} speaking in villa setting",
                camera_angle="medium_shot",
                mood="conversational",
                visual_elements=["villa interior"],
                character_emotions={msg.source: "engaged"},
                background_setting="villa_living_room",
                lighting="natural"
            ))
        
        return Storyboard(
            episode_title="Dating Show Episode",
            total_duration=len(segments) * 15.0,
            overall_mood="romantic",
            character_descriptions={},
            scene_segments=segments,
            transitions=["cut"],
            music_suggestions=["romantic_ambient"]
        )


# Convenience function for storyboard generation workflow
async def generate_storyboard_from_conversation(conversation_file: str, 
                                              model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]) -> str:
    """Workflow: Generate storyboard from conversation file."""
    generator = StoryboardGenerator(model_client)
    return await generator.generate_storyboard_from_conversation(conversation_file)