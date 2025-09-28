"""
Stage 3: Video Generation
This module handles the generation of video prompts and metadata from storyboard data using LLM.
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

from models import StoryboardData, VideoData, VideoSegment

load_dotenv(override=True)


class Stage3Pipeline:
    """Pipeline for Stage 3: Video generation from storyboards."""
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.video_output_dir = Path("generated_videos")
        self.video_output_dir.mkdir(exist_ok=True)
        self.model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash")
    
    def load_storyboard(self, filepath: str) -> StoryboardData:
        """Load storyboard data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return StoryboardData(**data)
    
    def _create_video_generation_prompt(self, storyboard_data: StoryboardData) -> str:
        """Create a detailed prompt for video generation."""
        
        # Extract scenes information
        scenes_info = []
        for scene in storyboard_data.scenes:
            scene_text = f"""
Scene {scene.scene_number}: {scene.title}
- Duration: {scene.duration_seconds}s
- Characters: {', '.join(scene.characters_involved)}
- Setting: {scene.setting}
- Mood: {scene.mood}
- Description: {scene.description}
- Key Dialogue: {scene.key_dialogue or 'None'}
- Visual Notes: {scene.visual_notes or 'None'}"""
            scenes_info.append(scene_text)
        
        # Get the JSON schema from the Pydantic model
        video_schema = VideoData.model_json_schema()
        
        prompt = f"""You are a professional video production AI specializing in creating detailed video generation prompts for AI video tools like Runway, Pika, or Sora.

Based on the following storyboard for a dating show episode, create detailed video generation prompts for each scene.

STORYBOARD DETAILS:
Title: {storyboard_data.title}
Total Duration: {storyboard_data.total_duration_seconds} seconds
Theme: {storyboard_data.overall_theme}
Target Audience: {storyboard_data.target_audience}

SCENES:
{chr(10).join(scenes_info)}

For each scene, create a video generation prompt that includes:
1. Detailed visual description
2. Camera movements and angles
3. Lighting and mood
4. Character actions and expressions
5. Setting and environment details
6. Style notes (cinematic, reality TV, documentary, etc.)
7. Technical specifications

The prompts should be optimized for AI video generation tools and create a cohesive, professional-looking dating show episode.

Make sure each video prompt is:
- Highly detailed and specific
- Optimized for AI video generation
- Consistent with reality TV/dating show aesthetics
- Technically feasible
- Emotionally engaging

Respond with a JSON object that matches this exact schema:

{json.dumps(video_schema, indent=2)}

Critical: make sure the JSON is valid and complete. Do not include any additional text outside the JSON response."""
        
        return prompt
    
    async def generate_video_prompts(self, storyboard_data: StoryboardData) -> VideoData:
        """Generate video prompts from storyboard data using LLM."""
        
        prompt = self._create_video_generation_prompt(storyboard_data)
        
        try:
            # Create message for the LLM
            message = UserMessage(content=prompt)
            
            # Get response from LLM
            response = await self.model_client.create([message])
            content = response.content
            
            # Parse JSON response
            try:
                video_dict = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {content}")
                raise ValueError("LLM response was not valid JSON")
            
            # Convert to VideoData model
            segments = []
            for segment_data in video_dict.get("segments", []):
                segment = VideoSegment(
                    segment_number=segment_data.get("segment_number", 0),
                    scene_reference=segment_data.get("scene_reference", 0),
                    video_prompt=segment_data.get("video_prompt", ""),
                    duration_seconds=float(segment_data.get("duration_seconds", 30.0)),
                    style_notes=segment_data.get("style_notes", ""),
                    characters=segment_data.get("characters", []),
                    setting=segment_data.get("setting", ""),
                    mood=segment_data.get("mood", "")
                )
                segments.append(segment)
            
            video_data = VideoData(
                title=video_dict.get("title", "Dating Show Episode"),
                total_duration_seconds=float(video_dict.get("total_duration_seconds", 300.0)),
                segments=segments,
                overall_style=video_dict.get("overall_style", "Reality TV style"),
                resolution=video_dict.get("resolution", "1920x1080"),
                fps=int(video_dict.get("fps", 24)),
                source_storyboard_file="",  # Will be set by caller
                metadata={
                    "pipeline_stage": 3,
                    "generation_method": "llm_video_prompts",
                    "model_used": "gemini-2.0-flash",
                    "storyboard_title": storyboard_data.title
                }
            )
            
            return video_data
            
        except Exception as e:
            print(f"Error generating video prompts: {e}")
            raise
    
    def save_video_data(self, video_data: VideoData, filename: str = None) -> str:
        """Save video data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_data_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(video_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Video data saved to: {filepath}")
        return str(filepath)
    
    def save_video_prompts_txt(self, video_data: VideoData, filename: str = None) -> str:
        """Save video prompts as a readable text file for manual use."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_prompts_{timestamp}.txt"
        
        filepath = self.video_output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"VIDEO GENERATION PROMPTS\n")
            f.write(f"========================\n\n")
            f.write(f"Title: {video_data.title}\n")
            f.write(f"Total Duration: {video_data.total_duration_seconds} seconds\n")
            f.write(f"Overall Style: {video_data.overall_style}\n")
            f.write(f"Resolution: {video_data.resolution}\n")
            f.write(f"FPS: {video_data.fps}\n\n")
            
            for segment in video_data.segments:
                f.write(f"SEGMENT {segment.segment_number} (Scene {segment.scene_reference})\n")
                f.write(f"Duration: {segment.duration_seconds}s\n")
                f.write(f"Characters: {', '.join(segment.characters)}\n")
                f.write(f"Setting: {segment.setting}\n")
                f.write(f"Mood: {segment.mood}\n")
                f.write(f"Style Notes: {segment.style_notes}\n\n")
                f.write(f"VIDEO PROMPT:\n")
                f.write(f"{segment.video_prompt}\n")
                f.write(f"\n{'='*50}\n\n")
        
        print(f"Video prompts text file saved to: {filepath}")
        return str(filepath)
    
    def print_video_summary(self, video_data: VideoData):
        """Print a human-readable summary of the video data."""
        print(f"\n=== VIDEO GENERATION DATA: {video_data.title} ===")
        print(f"Total Duration: {video_data.total_duration_seconds} seconds ({video_data.total_duration_seconds/60:.1f} minutes)")
        print(f"Overall Style: {video_data.overall_style}")
        print(f"Resolution: {video_data.resolution}")
        print(f"FPS: {video_data.fps}")
        print(f"Number of Segments: {len(video_data.segments)}")
        print("\n=== VIDEO SEGMENTS ===")
        
        for segment in video_data.segments:
            print(f"\nSegment {segment.segment_number} (Scene {segment.scene_reference})")
            print(f"  Duration: {segment.duration_seconds}s")
            print(f"  Characters: {', '.join(segment.characters)}")
            print(f"  Setting: {segment.setting}")
            print(f"  Mood: {segment.mood}")
            print(f"  Style Notes: {segment.style_notes}")
            print(f"  Video Prompt: {segment.video_prompt[:100]}...")
    
    async def run_full_stage3(self, storyboard_filepath: str) -> tuple[str, str]:
        """Run the complete Stage 3 pipeline."""
        try:
            # Load storyboard
            print(f"Loading storyboard from: {storyboard_filepath}")
            storyboard_data = self.load_storyboard(storyboard_filepath)
            
            # Generate video prompts
            print("Generating video prompts...")
            video_data = await self.generate_video_prompts(storyboard_data)
            video_data.source_storyboard_file = storyboard_filepath
            
            # Save video data
            video_data_filepath = self.save_video_data(video_data)
            
            # Save video prompts as text file
            video_prompts_filepath = self.save_video_prompts_txt(video_data)
            
            # Print summary
            self.print_video_summary(video_data)
            
            # Cleanup
            await self.model_client.close()
            
            return video_data_filepath, video_prompts_filepath
            
        except Exception as e:
            print(f"Error in Stage 3 pipeline: {e}")
            await self.model_client.close()
            raise


async def main():
    """Main function for running Stage 3 with CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate video prompts from storyboard data")
    parser.add_argument("--storyboard", required=True, help="Path to storyboard JSON file")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory for video data file")
    
    args = parser.parse_args()
    
    pipeline = Stage3Pipeline(output_dir=args.output_dir)
    
    try:
        video_data_filepath, video_prompts_filepath = await pipeline.run_full_stage3(args.storyboard)
        print(f"\nStage 3 completed successfully!")
        print(f"Video data file: {video_data_filepath}")
        print(f"Video prompts text file: {video_prompts_filepath}")
        print(f"\nYou can now use the video prompts with AI video generation tools like:")
        print(f"- Runway ML")
        print(f"- Pika Labs") 
        print(f"- OpenAI Sora")
        print(f"- Stable Video Diffusion")
        
    except Exception as e:
        print(f"Stage 3 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
