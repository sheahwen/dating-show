"""
Stage 2: Storyboard Generation
This module handles the generation of storyboards from conversation data using LLM.
"""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime
import re
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from dotenv import load_dotenv

from models import ConversationData, StoryboardData, StoryboardMetadata, StoryboardScene, StoryboardWithMetadataData

load_dotenv(override=True)

class Stage2Pipeline:
    """Pipeline for Stage 2: Storyboard generation from conversations."""
    
    def __init__(self, output_dir: str = "pipeline_output", config_path: str = "config/storyboarding_config.json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.model_client = self._create_model_client()
        self.storyboard_agent = self._create_storyboard_agent()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_model_client(self):
        """Create model client based on configuration."""
        model_config = self.config["model_client"]
        
        if model_config["client"] == "ollama":
            return OllamaChatCompletionClient(model=model_config["model"])
        elif model_config["client"] == "openai":
            return OpenAIChatCompletionClient(model=model_config["model"])
        else:
            raise ValueError(f"Unsupported client type: {model_config['client']}")
    
    def _create_storyboard_agent(self):
        """Create AssistantAgent with structured output for storyboard generation."""
        return AssistantAgent(
            name="storyboard_generator",
            model_client=self.model_client,
            system_message="You are a professional storyboard creator for dating shows. Generate detailed storyboards from conversation transcripts.",
            output_content_type=StoryboardData,  
        )
    
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
        
        # Get settings from config
        gen_settings = self.config["generation_settings"]
        prompt_settings = self.config["prompt_settings"]
        
        # Get the JSON schema from the Pydantic model
        storyboard_schema = StoryboardData.model_json_schema()
        
        prompt = f"""You are a professional TV show storyboard creator. Based on the following dating show conversation, create a detailed storyboard for a {gen_settings["target_duration_seconds"]}-second video segment.

PARTICIPANTS:
{chr(10).join(participants_info)}

DIRECTOR: {conversation_data.director.name}

CONVERSATION:
{chr(10).join(conversation_text)}

Create a storyboard with {gen_settings["min_scenes"]}-{gen_settings["max_scenes"]} scenes that captures the essence of this conversation. Each scene should be {gen_settings["scene_duration_range"]["min"]}-{gen_settings["scene_duration_range"]["max"]} seconds long.

For each scene, provide:
1. Scene number
2. Title (short, descriptive)
3. Detailed description of what happens
4. Duration in seconds
5. Characters involved
6. Setting/location
7. Mood/tone
8. Camera shot type (close-up, wide shot, medium shot, over-the-shoulder, establishing shot, etc.)
9. Key dialogue (if any)
10. Visual notes for filming

The storyboard should:
- Maintain the authentic flow of the conversation
- Highlight emotional moments and connections
- Show the dating show environment (villa, outdoor spaces, etc.)
- Create engaging television moments
- Include establishing shots and transitions

Respond with a JSON object that matches this exact schema:

{json.dumps(storyboard_schema, indent=2)}

CRITICAL INSTRUCTIONS:
- DO NOT return the schema above - create actual storyboard content
- Make sure the JSON is valid and complete. Do not include any additional text outside the JSON response."""
        
        return prompt
    
    async def generate_storyboard(self, conversation_data: ConversationData) -> StoryboardWithMetadataData:
        """Generate storyboard from conversation data using LLM."""
        
        prompt = self._create_storyboard_prompt(conversation_data)
        print(f"Storyboard prompt: {prompt}")
        
        try:
            result = await self.storyboard_agent.run(task=prompt)
            
            # The last message should be a StructuredMessage with StoryboardData content
            last_message = result.messages[-1]
            print(f"Agent response type: {type(last_message)}")
            
            if isinstance(last_message, StructuredMessage) and isinstance(last_message.content, StoryboardData):
                storyboard_data = last_message.content
                print(f"Successfully generated structured storyboard: {storyboard_data.title}")
                
                # Add metadata that's not in the LLM response
                storyboard_data_with_metadata = StoryboardWithMetadataData(
                    **storyboard_data.model_dump(),
                    metadata=StoryboardMetadata(
                        pipeline_stage=2,
                        generation_method="autogen_structured_output",
                        model_used=self.config["model_client"]["model"],
                        participants=[p.name for p in conversation_data.participants]
                    )
                )
                return storyboard_data_with_metadata
            else:
                raise ValueError("Storyboard agent response was not valid structured output")
            
        except Exception as e:
            print(f"Error generating storyboard: {e}")
            raise

    def extract_datetime_from_filepath(self, filepath: str) -> str:
        """Extract timestamp from filepath like 'pipeline_output/conversation_20250928_162628.json'"""
        pattern = r'conversation_(\d{8}_\d{6})\.json'
        match = re.search(pattern, filepath)
        if match:
            return match.group(1)  # Returns "20250928_162628"
        return None
    
    def save_storyboard(self, storyboard_data: StoryboardData, conversation_filename: str) -> str:
        """Save storyboard data to JSON file."""
        
        timestamp = self.extract_datetime_from_filepath(conversation_filename)
        base_filename = f"storyboard_{timestamp}"
        
        # check for existing files and add counter
        counter = 1
        filename = f"{base_filename}.json"
        while (self.output_dir / filename).exists():
            filename = f"{base_filename}_v{counter}.json"
            counter += 1

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
        print(f"Number of Scenes: {len(storyboard_data.scenes)}")
        print("\n=== SCENES ===")
        
        for scene in storyboard_data.scenes:
            print(f"\nScene {scene.scene_number}: {scene.title}")
            print(f"  Duration: {scene.duration_seconds}s")
            print(f"  Characters: {', '.join(scene.characters_involved)}")
            print(f"  Setting: {scene.setting}")
            print(f"  Mood: {scene.mood}")
            if scene.camera_shot:
                print(f"  Camera Shot: {scene.camera_shot}")
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
            storyboard_filepath = self.save_storyboard(storyboard_data, conversation_filepath)
            
            # Print summary
            # self.print_storyboard_summary(storyboard_data)
            
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
    parser.add_argument("--config", default="config/storyboarding_config.json", help="Path to storyboarding configuration file")
    
    args = parser.parse_args()
    
    pipeline = Stage2Pipeline(output_dir=args.output_dir, config_path=args.config)
    
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
