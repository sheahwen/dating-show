import asyncio
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from video_pipeline import Storyboard, SceneSegment


class VideoPromptGenerator:
    """Converts storyboard into detailed prompts for multimodal LLM video generation."""
    
    def __init__(self):
        self.output_dir = Path("video_prompts")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_video_prompts(self, storyboard: Storyboard) -> List[Dict[str, Any]]:
        """Generate detailed video prompts from storyboard."""
        video_prompts = []
        
        for i, segment in enumerate(storyboard.scene_segments):
            prompt = self._create_segment_prompt(segment, storyboard, i)
            video_prompts.append(prompt)
        
        # Save prompts to file for review
        self._save_prompts(video_prompts, storyboard.episode_title)
        
        return video_prompts
    
    def _create_segment_prompt(self, segment: SceneSegment, 
                             storyboard: Storyboard, segment_index: int) -> Dict[str, Any]:
        """Create detailed prompt for a single video segment."""
        
        # Get character descriptions
        char_desc = storyboard.character_descriptions.get(segment.speaker, "person")
        
        # Build comprehensive prompt
        visual_prompt = f"""
        SCENE {segment_index + 1}: {segment.scene_description}

        CHARACTER: {char_desc}
        DIALOGUE: "{segment.dialogue}"
        SETTING: {segment.background_setting}
        MOOD: {segment.mood}
        LIGHTING: {segment.lighting}
        CAMERA: {segment.camera_angle}
        
        VISUAL DETAILS:
        - {', '.join(segment.visual_elements)}
        
        CHARACTER EMOTIONS:
        {', '.join([f"{char}: {emotion}" for char, emotion in segment.character_emotions.items()])}
        
        STYLE: Reality TV, cinematic, {storyboard.overall_mood} atmosphere
        DURATION: {segment.duration_seconds} seconds
        
        Generate a video showing this scene with natural lip-sync for the dialogue, 
        appropriate facial expressions, and {segment.camera_angle} camera work.
        """
        
        return {
            "segment_index": segment_index,
            "speaker": segment.speaker,
            "dialogue": segment.dialogue,
            "duration": segment.duration_seconds,
            "visual_prompt": visual_prompt.strip(),
            "style_params": {
                "mood": segment.mood,
                "lighting": segment.lighting,
                "camera_angle": segment.camera_angle,
                "background": segment.background_setting
            },
            "audio_prompt": f"Natural speech: '{segment.dialogue}' spoken by {char_desc} with {segment.character_emotions.get(segment.speaker, 'neutral')} emotion"
        }
    
    def _save_prompts(self, prompts: List[Dict], episode_title: str) -> None:
        """Save generated prompts to file for review."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{episode_title.replace(' ', '_')}_{timestamp}_prompts.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"📝 Video prompts saved to: {filepath}")


class MultimodalVideoGenerator:
    """Stage 3: Generates videos using multimodal LLM (Runway, Pika, etc.)."""
    
    def __init__(self, provider: str = "runway"):
        self.provider = provider
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)
    
    async def generate_videos_from_storyboard(self, storyboard_file: str) -> Dict[str, Any]:
        """Generate videos from storyboard file."""
        print(f"🎥 Loading storyboard from: {storyboard_file}")
        
        # Load storyboard
        storyboard = Storyboard.load_from_file(storyboard_file)
        print(f"✅ Loaded storyboard: {storyboard.episode_title}")
        print(f"📊 {len(storyboard.scene_segments)} segments, {storyboard.total_duration}s total")
        
        # Generate video prompts
        prompt_generator = VideoPromptGenerator()
        video_prompts = prompt_generator.generate_video_prompts(storyboard)
        
        # Generate videos
        video_files = await self.generate_videos_from_prompts(video_prompts)
        
        # Create episode folder
        episode_folder = self.output_dir / f"{storyboard.episode_title.replace(' ', '_')}"
        episode_folder.mkdir(exist_ok=True)
        
        # Compile results
        results = {
            "storyboard_file": storyboard_file,
            "storyboard": storyboard,
            "video_prompts": video_prompts,
            "video_files": video_files,
            "episode_folder": str(episode_folder),
            "total_segments": len(video_prompts),
            "total_duration": storyboard.total_duration
        }
        
        # Save results summary
        results_file = episode_folder / "generation_results.json"
        with open(results_file, 'w') as f:
            # Convert storyboard to dict for JSON serialization
            results_copy = results.copy()
            results_copy["storyboard"] = storyboard.model_dump()
            json.dump(results_copy, f, indent=2)
        
        print(f"\n🎉 VIDEO GENERATION COMPLETE!")
        print(f"📊 Generated {len(video_files)} video segments")
        print(f"⏱️ Total duration: {storyboard.total_duration}s")
        print(f"📁 Episode folder: {episode_folder}")
        print(f"📄 Results saved to: {results_file}")
        
        return results
    
    async def generate_videos_from_prompts(self, prompts: List[Dict[str, Any]]) -> List[str]:
        """Generate video files from prompts."""
        video_files = []
        
        for prompt in prompts:
            print(f"🎬 Generating video for segment {prompt['segment_index'] + 1}...")
            
            try:
                video_file = await self._generate_single_video(prompt)
                video_files.append(video_file)
                print(f"✅ Generated: {video_file}")
                
            except Exception as e:
                print(f"❌ Error generating segment {prompt['segment_index']}: {e}")
                # Create placeholder for failed generation
                video_files.append(None)
        
        return video_files
    
    async def _generate_single_video(self, prompt: Dict[str, Any]) -> str:
        """Generate a single video segment."""
        if self.provider == "runway":
            return await self._generate_runway_video(prompt)
        elif self.provider == "pika":
            return await self._generate_pika_video(prompt)
        elif self.provider == "mock":
            return await self._generate_mock_video(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_runway_video(self, prompt: Dict[str, Any]) -> str:
        """Generate video using Runway ML API."""
        # Placeholder for Runway API integration
        # You would implement actual API calls here
        
        print(f"🎯 Runway prompt: {prompt['visual_prompt'][:100]}...")
        
        # Simulate API call delay
        await asyncio.sleep(2)
        
        # Return mock filename
        filename = f"runway_segment_{prompt['segment_index']}.mp4"
        return str(self.output_dir / filename)
    
    async def _generate_pika_video(self, prompt: Dict[str, Any]) -> str:
        """Generate video using Pika Labs API."""
        # Placeholder for Pika API integration
        
        print(f"🎯 Pika prompt: {prompt['visual_prompt'][:100]}...")
        
        # Simulate API call delay
        await asyncio.sleep(3)
        
        filename = f"pika_segment_{prompt['segment_index']}.mp4"
        return str(self.output_dir / filename)
    
    async def _generate_mock_video(self, prompt: Dict[str, Any]) -> str:
        """Generate mock video for testing."""
        print(f"🎯 Mock generation: {prompt['visual_prompt'][:100]}...")
        
        # Create a simple text file as placeholder
        filename = f"mock_segment_{prompt['segment_index']}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"Mock video segment {prompt['segment_index']}\n")
            f.write(f"Speaker: {prompt['speaker']}\n")
            f.write(f"Dialogue: {prompt['dialogue']}\n")
            f.write(f"Duration: {prompt['duration']}s\n")
            f.write(f"Visual: {prompt['visual_prompt']}\n")
        
        return str(filepath)


# Convenience function for video generation workflow
async def generate_videos_from_storyboard(storyboard_file: str, provider: str = "mock") -> Dict[str, Any]:
    """Workflow: Generate videos from storyboard file."""
    generator = MultimodalVideoGenerator(provider)
    return await generator.generate_videos_from_storyboard(storyboard_file)
