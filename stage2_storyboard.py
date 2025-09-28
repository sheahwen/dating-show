"""
Stage 2: Storyboard Generation
This module handles the generation of storyboards from conversation data using LLM.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from dotenv import load_dotenv

from models import ConversationData, StoryboardData, StoryboardScene

load_dotenv(override=True)


class Stage2Pipeline:
    """Pipeline for Stage 2: Storyboard generation from conversations."""
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash")
    
    def load_conversation(self, filepath: str) -> ConversationData:
        """Load conversation data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ConversationData(**data)
    
    def _create_storyboard_prompt(self, conversation_data: ConversationData) -> str:
        """Create a detailed prompt for storyboard generation."""
        
        # Extract participant names and traits
        participants_info = []
        for participant in conversation_data.participants:
            traits = ", ".join(participant.personality_traits)
            participants_info.append(f"- {participant.name} ({participant.age}): {traits}")
        
        # Extract key conversation moments
        conversation_text = []
        for msg in conversation_data.messages:
            if msg.message_type.value != "user":  # Skip system messages
                conversation_text.append(f"{msg.source}: {msg.content}")
        
        # Get the JSON schema from the Pydantic model
        storyboard_schema = StoryboardData.model_json_schema()
        
        prompt = f"""You are a professional TV show storyboard creator. Based on the following dating show conversation, create a detailed storyboard for a 5-minute video segment.

PARTICIPANTS:
{chr(10).join(participants_info)}

DIRECTOR: {conversation_data.director.name}

CONVERSATION:
{chr(10).join(conversation_text)}

Create a storyboard with 8-12 scenes that captures the essence of this conversation. Each scene should be 20-40 seconds long.

For each scene, provide:
1. Scene number
2. Title (short, descriptive)
3. Detailed description of what happens
4. Duration in seconds
5. Characters involved
6. Setting/location
7. Mood/tone
8. Key dialogue (if any)
9. Visual notes for filming

The storyboard should:
- Maintain the authentic flow of the conversation
- Highlight emotional moments and connections
- Include reaction shots and close-ups
- Show the dating show environment (villa, outdoor spaces, etc.)
- Create engaging television moments
- Build romantic tension and drama
- Include establishing shots and transitions

Respond with a JSON object that matches this exact schema:

{json.dumps(storyboard_schema, indent=2)}

Critical: make sure the JSON is valid and complete. Do not include any additional text outside the JSON response."""
        
        return prompt
    
    async def generate_storyboard(self, conversation_data: ConversationData) -> StoryboardData:
        """Generate storyboard from conversation data using LLM."""
        
        prompt = self._create_storyboard_prompt(conversation_data)
        
        try:
            # Create message for the LLM
            message = UserMessage(content=prompt)
            
            # Get response from LLM
            response = await self.model_client.create([message])
            content = response.content
            
            # Parse JSON response
            try:
                storyboard_dict = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {content}")
                raise ValueError("LLM response was not valid JSON")
            
            # Convert to StoryboardData model
            scenes = []
            for scene_data in storyboard_dict.get("scenes", []):
                scene = StoryboardScene(
                    scene_number=scene_data.get("scene_number", 0),
                    title=scene_data.get("title", ""),
                    description=scene_data.get("description", ""),
                    duration_seconds=float(scene_data.get("duration_seconds", 30.0)),
                    characters_involved=scene_data.get("characters_involved", []),
                    setting=scene_data.get("setting", ""),
                    mood=scene_data.get("mood", ""),
                    key_dialogue=scene_data.get("key_dialogue"),
                    visual_notes=scene_data.get("visual_notes")
                )
                scenes.append(scene)
            
            storyboard_data = StoryboardData(
                title=storyboard_dict.get("title", "Dating Show Episode"),
                total_duration_seconds=float(storyboard_dict.get("total_duration_seconds", 300.0)),
                scenes=scenes,
                overall_theme=storyboard_dict.get("overall_theme", ""),
                target_audience=storyboard_dict.get("target_audience", "Young adults"),
                source_conversation_file="",  # Will be set by caller
                metadata={
                    "pipeline_stage": 2,
                    "generation_method": "llm_storyboard",
                    "model_used": "gemini-2.0-flash",
                    "participants": [p.name for p in conversation_data.participants]
                }
            )
            
            return storyboard_data
            
        except Exception as e:
            print(f"Error generating storyboard: {e}")
            raise
    
    def save_storyboard(self, storyboard_data: StoryboardData, filename: str = None) -> str:
        """Save storyboard data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"storyboard_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(storyboard_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Storyboard saved to: {filepath}")
        return str(filepath)
    
    def print_storyboard_summary(self, storyboard_data: StoryboardData):
        """Print a human-readable summary of the storyboard."""
        print(f"\n=== STORYBOARD: {storyboard_data.title} ===")
        print(f"Total Duration: {storyboard_data.total_duration_seconds} seconds ({storyboard_data.total_duration_seconds/60:.1f} minutes)")
        print(f"Theme: {storyboard_data.overall_theme}")
        print(f"Target Audience: {storyboard_data.target_audience}")
        print(f"Number of Scenes: {len(storyboard_data.scenes)}")
        print("\n=== SCENES ===")
        
        for scene in storyboard_data.scenes:
            print(f"\nScene {scene.scene_number}: {scene.title}")
            print(f"  Duration: {scene.duration_seconds}s")
            print(f"  Characters: {', '.join(scene.characters_involved)}")
            print(f"  Setting: {scene.setting}")
            print(f"  Mood: {scene.mood}")
            print(f"  Description: {scene.description}")
            if scene.key_dialogue:
                print(f"  Key Dialogue: {scene.key_dialogue}")
            if scene.visual_notes:
                print(f"  Visual Notes: {scene.visual_notes}")
    
    async def run_full_stage2(self, conversation_filepath: str) -> str:
        """Run the complete Stage 2 pipeline."""
        try:
            # Load conversation
            print(f"Loading conversation from: {conversation_filepath}")
            conversation_data = self.load_conversation(conversation_filepath)
            
            # Generate storyboard
            print("Generating storyboard...")
            storyboard_data = await self.generate_storyboard(conversation_data)
            storyboard_data.source_conversation_file = conversation_filepath
            
            # Save storyboard
            storyboard_filepath = self.save_storyboard(storyboard_data)
            
            # Print summary
            self.print_storyboard_summary(storyboard_data)
            
            # Cleanup
            await self.model_client.close()
            
            return storyboard_filepath
            
        except Exception as e:
            print(f"Error in Stage 2 pipeline: {e}")
            await self.model_client.close()
            raise


async def main():
    """Main function for running Stage 2 with CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate storyboard from conversation data")
    parser.add_argument("--conversation", required=True, help="Path to conversation JSON file")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory for storyboard file")
    
    args = parser.parse_args()
    
    pipeline = Stage2Pipeline(output_dir=args.output_dir)
    
    try:
        storyboard_filepath = await pipeline.run_full_stage2(args.conversation)
        print(f"\nStage 2 completed successfully!")
        print(f"Storyboard file: {storyboard_filepath}")
        print(f"\nTo proceed to Stage 3, run:")
        print(f"python stage3_video.py --storyboard {storyboard_filepath}")
        
    except Exception as e:
        print(f"Stage 2 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
