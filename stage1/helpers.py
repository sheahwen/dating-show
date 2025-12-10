"""Stage 1 helper utilities to keep the pipeline orchestrator lean."""

from pathlib import Path
from typing import List

from common.io import save_json, timestamped_filename
from common.models import (
    ConversationData,
    ConversationMessage,
    DirectorInfo,
    MessageType,
    ParticipantInfo,
)
from common.text import extract_content_without_thoughts
from stage1.agents import DirectorAgent, ParticipantAgent


def generate_base_filename(prefix: str = "conversation") -> str:
    """Return a timestamped base filename without extension."""
    return timestamped_filename(prefix).replace(".json", "")


def convert_messages_to_conversation_data(
    messages,
    participants: List[ParticipantAgent],
    director: DirectorAgent,
    max_turns: int,
    total_turns: int,
) -> ConversationData:
    """Convert raw messages to ConversationData model."""
    participant_infos = [
        ParticipantInfo(
            name=p.name,
            age=p.age,
            personality_traits=p.personality_traits,
            system_message=p.get_system_message(),
        )
        for p in participants
    ]

    director_info = DirectorInfo(
        name=director.name, system_message=director.get_system_message()
    )

    conversation_messages: list[ConversationMessage] = []
    for msg in messages:
        if msg.source == "producer":
            msg_type = MessageType.USER
        elif msg.source == "director":
            msg_type = MessageType.DIRECTOR
        elif msg.source in [p.name for p in participants]:
            msg_type = MessageType.PARTICIPANT
        else:
            msg_type = MessageType.ASSISTANT

        conversation_messages.append(
            ConversationMessage(
                content=extract_content_without_thoughts(msg.content),
                source=msg.source,
                message_type=msg_type,
            )
        )

    return ConversationData(
        participants=participant_infos,
        director=director_info,
        messages=conversation_messages,
        max_turns=max_turns,
        total_turns=total_turns,
        metadata={"pipeline_stage": 1, "generation_method": "autogen_agents"},
    )


def save_conversation(
    conversation_data: ConversationData, output_dir: Path, filename: str | None = None
) -> str:
    """Save conversation data to JSON file."""
    if filename is None:
        filename = timestamped_filename("conversation")

    filepath = output_dir / filename
    save_json(conversation_data.model_dump(), filepath)
    return str(filepath)


def save_checkpoint(
    conversation_data: ConversationData,
    output_dir: Path,
    base_filename: str | None,
    checkpoint_counter: int,
    is_final: bool = False,
) -> tuple[str, int]:
    """Save conversation checkpoint with incremental naming and return new counter."""
    base_filename = base_filename or generate_base_filename("conversation")

    if base_filename.endswith(".json"):
        base_filename = base_filename[:-5]

    if is_final:
        filename = f"{base_filename}_final.json"
    else:
        checkpoint_counter += 1
        filename = f"{base_filename}_checkpoint_{checkpoint_counter:03d}.json"

    filepath = output_dir / filename
    save_json(conversation_data.model_dump(), filepath)
    return str(filepath), checkpoint_counter
