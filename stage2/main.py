"""
Stage 2: Storyboard Generation
This module handles the generation of storyboards from conversation data using LLM.
"""

import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_core.models import UserMessage

from common.clients import create_model_client
from common.io import extract_timestamp_from_filepath, save_json
from common.models import (
    ConversationData,
    StoryboardData,
    StoryboardMetadata,
    StoryboardScene,
    StoryboardWithMetadataData,
)
from stage2.prompts import build_storyboard_prompt, summarize_highlights

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
        return create_model_client(model_config)
    
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
    
    async def generate_storyboard(self, conversation_data: ConversationData) -> StoryboardWithMetadataData:
        """Generate storyboard from conversation data using LLM."""
        highlight_limit = self.config.get("highlight_settings", {}).get("max_highlights", 6)
        highlights = summarize_highlights(conversation_data, max_highlights=highlight_limit)
        prompt = build_storyboard_prompt(conversation_data, self.config, highlights)
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

    def save_storyboard(self, storyboard_data: StoryboardData, conversation_filename: str) -> str:
        """Save storyboard data to JSON file."""
        timestamp = extract_timestamp_from_filepath(conversation_filename, prefix="conversation")
        base_filename = f"storyboard_{timestamp}"
        
        # check for existing files and add counter
        counter = 1
        filename = f"{base_filename}.json"
        while (self.output_dir / filename).exists():
            filename = f"{base_filename}_v{counter}.json"
            counter += 1

        filepath = self.output_dir / filename
        
        save_json(storyboard_data.model_dump(), filepath)

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
