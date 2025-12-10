"""Memory management for dating show agents."""

from typing import List
from autogen_agentchat.messages import TextMessage

from stage1.agents.director import DirectorAgent
from stage1.agents.participant import ParticipantAgent


class MemoryManager:
    """Handles memory updates for all agents in the dating show."""

    def __init__(self, participants: List[ParticipantAgent], director: DirectorAgent):
        self.participants = participants
        self.director = director

    async def update_all_memories(
        self, recent_messages: List[TextMessage], turn_count: int
    ) -> None:
        """Update memory for all agents based on recent conversation."""
        try:
            participant_names = [p.name for p in self.participants]

            # Update participant memories
            for participant in self.participants:
                await participant.update_memory(recent_messages, participant_names)

            # Update director memory
            await self.director.update_memory(recent_messages, participant_names)

            print(f"🧠 Updated memories for all agents (turn {turn_count})")

        except Exception as e:
            print(f"Error updating agent memories: {e}")
