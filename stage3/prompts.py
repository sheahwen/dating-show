"""Prompt helpers for Stage 3 video generation."""

import json

from common.models import StoryboardData, VideoData


def build_video_generation_prompt(storyboard_data: StoryboardData) -> str:
    """Create a detailed prompt for turning storyboard scenes into video prompts."""
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

    video_schema = VideoData.model_json_schema()

    return f"""You are a professional video production AI specializing in creating detailed video generation prompts for AI video tools like Runway, Pika, Sora, or Veo.

Based on the following storyboard for a dating show episode, create detailed video generation prompts for each scene.

STORYBOARD DETAILS:
Title: {storyboard_data.title}
Total Duration: {storyboard_data.total_duration_seconds} seconds
Theme: {storyboard_data.overall_theme}

SCENES:
{chr(10).join(scenes_info)}

For each scene, create a video generation prompt that includes:
1. Detailed visual description
2. Camera shot type (close-up, wide shot, medium shot, over-the-shoulder, establishing shot, etc.)
3. Camera movements and angles
4. Lighting and mood
5. Character actions and expressions
6. Setting and environment details
7. Style notes (cinematic, reality TV, documentary, etc.)
8. Technical specifications

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

