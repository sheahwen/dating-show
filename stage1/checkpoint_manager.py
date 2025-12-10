"""Checkpoint management for the dating show conversation."""

from typing import List, Optional

from stage1.agents.director import DirectorAgent
from stage1.agents.participant import ParticipantAgent
from stage1.message_manager import MessageManager


class CheckpointManager:
    """Handles saving conversation checkpoints."""

    def __init__(
        self,
        participants: List[ParticipantAgent],
        director: DirectorAgent,
        max_turns: int,
        save_frequency: int = 10,
        pipeline=None,  # Stage1Pipeline reference for saving
        base_filename: Optional[str] = None,
    ):
        self.participants = participants
        self.director = director
        self.max_turns = max_turns
        self.save_frequency = save_frequency
        self.pipeline = pipeline
        self.base_filename = base_filename

    async def save_checkpoint_if_needed(
        self, message_manager: MessageManager, turn_count: int
    ) -> None:
        """Save conversation checkpoint if pipeline is available and it's time to save."""
        if self.pipeline and turn_count > 0 and turn_count % self.save_frequency == 0:
            try:
                participant_infos = [
                    {
                        "name": p.name,
                        "age": p.age,
                        "personality_traits": p.personality_traits,
                        "system_message": p.get_system_message(),
                    }
                    for p in self.participants
                ]

                director_info = {
                    "name": self.director.name,
                    "system_message": self.director.get_system_message(),
                }

                conversation_messages = []
                for msg in message_manager.get_original_messages():
                    from common.text import extract_content_without_thoughts
                    from common.models import MessageType

                    if msg.source == "producer":
                        msg_type = MessageType.USER
                    elif msg.source == "director":
                        msg_type = MessageType.DIRECTOR
                    elif msg.source in [p.name for p in self.participants]:
                        msg_type = MessageType.PARTICIPANT
                    else:
                        msg_type = MessageType.ASSISTANT

                    conversation_msg = {
                        "content": extract_content_without_thoughts(msg.content),
                        "source": msg.source,
                        "message_type": msg_type.value,
                    }
                    conversation_messages.append(conversation_msg)

                from common.models import (
                    ConversationData,
                    ConversationMessage,
                    DirectorInfo,
                    ParticipantInfo,
                )

                participant_info_objects = [
                    ParticipantInfo(**info) for info in participant_infos
                ]
                director_info_object = DirectorInfo(**director_info)
                conversation_message_objects = [
                    ConversationMessage(**msg) for msg in conversation_messages
                ]

                conversation_data = ConversationData(
                    participants=participant_info_objects,
                    director=director_info_object,
                    messages=conversation_message_objects,
                    max_turns=self.max_turns,
                    total_turns=turn_count,
                    metadata={
                        "pipeline_stage": 1,
                        "generation_method": "autogen_agents",
                        "checkpoint": True,
                    },
                )

                self.pipeline.save_checkpoint(conversation_data, self.base_filename)

            except Exception as e:
                print(f"Error saving checkpoint at turn {turn_count}: {e}")
