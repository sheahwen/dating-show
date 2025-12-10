"""Prompt and highlight helpers for Stage 2 storyboard generation."""

import json
from typing import List

from common.models import ConversationData, StoryboardData, StoryboardScene


def summarize_highlights(
    conversation_data: ConversationData, max_highlights: int = 6
) -> List[str]:
    """
    Select conversational highlights to feed into storyboard generation.
    Simple heuristic: pick the last N non-system messages from participants/director.
    """
    relevant = [
        m
        for m in conversation_data.messages
        if m.message_type.value in {"participant", "director", "assistant"}
    ]
    trimmed = relevant[-max_highlights:]
    return [f"{m.source}: {m.content}" for m in trimmed]


def _format_participants(conversation_data: ConversationData) -> str:
    parts = []
    for participant in conversation_data.participants:
        traits = ", ".join(participant.personality_traits)
        parts.append(f"- {participant.name} ({participant.age}): {traits}")
    return "\n".join(parts)


def build_storyboard_prompt(
    conversation_data: ConversationData, config: dict, highlights: List[str]
) -> str:
    """Compose the storyboard prompt using highlights instead of full transcript."""
    gen_settings = config["generation_settings"]
    storyboard_schema = StoryboardData.model_json_schema()

    highlight_text = (
        "\n".join(highlights)
        if highlights
        else "No highlights were detected; create a concise storyboard."
    )

    prompt = f"""You are a professional TV show storyboard creator. Based on the following dating show highlights, create a detailed storyboard for a {gen_settings["target_duration_seconds"]}-second video segment.

PARTICIPANTS:
{_format_participants(conversation_data)}

DIRECTOR: {conversation_data.director.name}

CONVERSATION HIGHLIGHTS:
{highlight_text}

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
