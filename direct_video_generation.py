#!/usr/bin/env python3
"""
Direct Video Generation from Storyboard
This script converts storyboard data directly to video generation prompts without the intermediate Stage 3 step.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

from models import StoryboardData, StoryboardScene


class DirectVideoGenerator:
    """Convert storyboard data directly to video generation prompts."""
    
    def __init__(self, output_dir: str = "generated_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_storyboard(self, filepath: str) -> StoryboardData:
        """Load storyboard data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return StoryboardData(**data)
    
    def scene_to_video_prompt(self, scene: StoryboardScene) -> str:
        """Convert a single storyboard scene to a video generation prompt."""
        
        # Build the core prompt from scene data
        prompt_parts = []
        
        # Basic scene setup
        prompt_parts.append(f"Scene: {scene.description}")
        
        # Setting and environment
        prompt_parts.append(f"Setting: {scene.setting}")
        
        # Characters and their actions
        if scene.characters_involved:
            characters_str = ", ".join(scene.characters_involved)
            prompt_parts.append(f"Characters: {characters_str}")
        
        # Camera work
        if scene.camera_shot:
            prompt_parts.append(f"Camera: {scene.camera_shot}")
        
        # Mood and atmosphere
        prompt_parts.append(f"Mood: {scene.mood}")
        
        # Visual details
        if scene.visual_notes:
            prompt_parts.append(f"Visual details: {scene.visual_notes}")
        
        # Dialogue context
        if scene.key_dialogue:
            prompt_parts.append(f"Key dialogue: \"{scene.key_dialogue}\"")
        
        # Technical specs for AI video tools
        prompt_parts.append(f"Duration: {scene.duration_seconds} seconds")
        prompt_parts.append("Style: Reality TV, dating show aesthetic, professional cinematography")
        prompt_parts.append("Quality: High definition, cinematic lighting, smooth camera movements")
        
        return ". ".join(prompt_parts) + "."
    
    def generate_video_prompts_file(self, storyboard_data: StoryboardData, filename: str = None) -> str:
        """Generate a text file with video prompts for each scene."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_prompts_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("AI VIDEO GENERATION PROMPTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Episode: {storyboard_data.title}\n")
            f.write(f"Total Duration: {storyboard_data.total_duration_seconds} seconds\n")
            f.write(f"Theme: {storyboard_data.overall_theme}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INDIVIDUAL SCENE PROMPTS:\n")
            f.write("-" * 30 + "\n\n")
            
            for scene in storyboard_data.scenes:
                f.write(f"SCENE {scene.scene_number}: {scene.title}\n")
                f.write(f"Duration: {scene.duration_seconds}s\n\n")
                
                video_prompt = self.scene_to_video_prompt(scene)
                f.write("VIDEO PROMPT:\n")
                f.write(video_prompt)
                f.write("\n\n" + "=" * 50 + "\n\n")
            
            # Also provide a single combined prompt for the entire episode
            f.write("COMBINED EPISODE PROMPT:\n")
            f.write("-" * 25 + "\n\n")
            
            combined_scenes = []
            for scene in storyboard_data.scenes:
                scene_summary = f"Scene {scene.scene_number} ({scene.duration_seconds}s): {scene.description} in {scene.setting} with {scene.mood} mood"
                if scene.camera_shot:
                    scene_summary += f", filmed with {scene.camera_shot}"
                combined_scenes.append(scene_summary)
            
            combined_prompt = f"Create a {storyboard_data.total_duration_seconds}-second dating show episode titled '{storyboard_data.title}' with theme '{storyboard_data.overall_theme}'. " + "; ".join(combined_scenes) + ". Style: Reality TV, professional cinematography, high definition."
            
            f.write(combined_prompt)
            f.write("\n\n")
        
        print(f"Video prompts saved to: {filepath}")
        return str(filepath)
    
    def generate_json_prompts(self, storyboard_data: StoryboardData, filename: str = None) -> str:
        """Generate a JSON file with structured video prompts."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_prompts_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        video_data = {
            "title": storyboard_data.title,
            "total_duration_seconds": storyboard_data.total_duration_seconds,
            "overall_theme": storyboard_data.overall_theme,
            "generated_at": datetime.now().isoformat(),
            "source_storyboard": getattr(storyboard_data, 'source_conversation_file', 'unknown'),
            "scenes": []
        }
        
        for scene in storyboard_data.scenes:
            scene_data = {
                "scene_number": scene.scene_number,
                "title": scene.title,
                "duration_seconds": scene.duration_seconds,
                "video_prompt": self.scene_to_video_prompt(scene),
                "characters": scene.characters_involved,
                "setting": scene.setting,
                "mood": scene.mood,
                "camera_shot": scene.camera_shot,
                "original_description": scene.description
            }
            video_data["scenes"].append(scene_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON video prompts saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self, storyboard_data: StoryboardData):
        """Print a summary of the video generation process."""
        print(f"\n=== DIRECT VIDEO GENERATION ===")
        print(f"Episode: {storyboard_data.title}")
        print(f"Duration: {storyboard_data.total_duration_seconds}s ({storyboard_data.total_duration_seconds/60:.1f} minutes)")
        print(f"Scenes: {len(storyboard_data.scenes)}")
        print(f"Theme: {storyboard_data.overall_theme}")
        
        print(f"\n=== SCENE BREAKDOWN ===")
        for scene in storyboard_data.scenes:
            print(f"Scene {scene.scene_number}: {scene.title} ({scene.duration_seconds}s)")
            print(f"  → {scene.description[:80]}...")
    
    def process_storyboard(self, storyboard_filepath: str) -> tuple[str, str]:
        """Process a storyboard file and generate video prompts."""
        
        print(f"Loading storyboard: {storyboard_filepath}")
        storyboard_data = self.load_storyboard(storyboard_filepath)
        
        # Generate both text and JSON formats
        text_file = self.generate_video_prompts_file(storyboard_data)
        json_file = self.generate_json_prompts(storyboard_data)
        
        self.print_summary(storyboard_data)
        
        return text_file, json_file


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Generate video prompts directly from storyboard data")
    parser.add_argument("--storyboard", required=True, help="Path to storyboard JSON file")
    parser.add_argument("--output-dir", default="generated_videos", help="Output directory for video prompt files")
    
    args = parser.parse_args()
    
    generator = DirectVideoGenerator(output_dir=args.output_dir)
    
    try:
        text_file, json_file = generator.process_storyboard(args.storyboard)
        
        print(f"\n✅ SUCCESS! Video prompts generated:")
        print(f"📄 Text format: {text_file}")
        print(f"📋 JSON format: {json_file}")
        print(f"\n🎬 Ready for AI video generation tools:")
        print(f"   • Runway ML")
        print(f"   • Pika Labs") 
        print(f"   • OpenAI Sora")
        print(f"   • Stable Video Diffusion")
        print(f"   • Any other AI video tool")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
