"""Dating show chat orchestrator."""

from typing import List
from autogen_core.models import LLMMessage, UserMessage

from common.text import extract_content_without_thoughts
from stage1 import DirectorAgent, ParticipantAgent
from stage1.prompts import INITIAL_PRODUCER_MESSAGE
from stage1.message_manager import MessageManager
from stage1.memory_manager import MemoryManager
from stage1.conversation_controller import ConversationController
from stage1.checkpoint_manager import CheckpointManager


class DatingShowChat:
    """Chat orchestrator for the dating show."""

    def __init__(
        self,
        participants: List[ParticipantAgent],
        director: DirectorAgent,
        max_turns: int = 5,
        director_intervention_probability: float = 0.05,
        save_frequency: int = 10,
        pipeline=None,  # Stage1Pipeline reference for saving
        base_filename: str = None,
    ):
        """
        Initialize the dating show chat.

        Args:
            participants: List of participant agents
            director: Director agent
            max_turns: Maximum number of turns before termination
            director_intervention_probability: Probability of director intervention
            pipeline: Stage1Pipeline reference for incremental saving
            base_filename: Base filename for checkpoint saves
        """
        self.participants = participants
        self.director = director
        self.max_turns = max_turns
        self.turn_count = 0
        self.memory_update_frequency = 3
        self.message_manager = MessageManager()
        self.memory_manager = MemoryManager(participants, director)
        self.conversation_controller = ConversationController(
            participants, director, director_intervention_probability
        )
        self.checkpoint_manager = CheckpointManager(
            participants, director, max_turns, save_frequency, pipeline, base_filename
        )

    async def run(self) -> List[LLMMessage]:
        """Run the conversation."""
        print(f"=== Starting Dating Show Conversation ===")

        if self.message_manager.get_message_count() == 0:
            initial_message = UserMessage(
                content=INITIAL_PRODUCER_MESSAGE, source="producer"
            )
            self.message_manager.add_message(initial_message)

            director_intro = self.director.get_director_intro()
            print(f"Director: {director_intro}")
            self.message_manager.add_message(director_intro)
            self.turn_count += 1

        recent_messages = self.message_manager.get_recent_messages(limit=10)
        await self.memory_manager.update_all_memories(recent_messages, self.turn_count)

        while self.turn_count < self.max_turns:
            print("\n" + "=" * 50)
            print("🎭 Selecting next speaker based on interest...")

            next_speaker = await self.conversation_controller.select_next_speaker(
                recent_messages
            )

            if not next_speaker:
                print("No participants want to speak. Conversation ending.")
                break

            print(f"🎤 {next_speaker.name} wants to speak!")
            print("\n" + "=" * 50)
            print(f"Participant {next_speaker.name}'s turn to speak \n")

            participant_response = await next_speaker._get_participant_response(
                recent_messages
            )
            response_content = extract_content_without_thoughts(
                participant_response.chat_message.content
            )
            print(f"Participant {next_speaker.name}: {response_content}")

            if response_content.strip().replace(".", "").isdigit():
                print(
                    f"⚠️  WARNING: {next_speaker.name} gave a numeric response instead of conversation: '{response_content}'"
                )

            self.message_manager.add_message(participant_response.chat_message)
            self.conversation_controller.increment_speak_count(next_speaker.name)
            self.turn_count += 1

            if self.turn_count % self.memory_update_frequency == 0:
                recent_messages = self.message_manager.get_recent_messages(limit=10)
                await self.memory_manager.update_all_memories(
                    recent_messages, self.turn_count
                )

            await self.checkpoint_manager.save_checkpoint_if_needed(
                self.message_manager, self.turn_count
            )

            if self.turn_count >= self.max_turns:
                break

            recent_messages = self.message_manager.get_recent_messages(limit=8)
            should_intervene = (
                await self.conversation_controller.check_director_intervention(
                    recent_messages
                )
            )

            if should_intervene:
                print("\n" + "=" * 50)
                print("Director intervention triggered!")
                director_response = await self.director.get_director_intervention(
                    recent_messages
                )

                if director_response:
                    print(
                        f"Director: {extract_content_without_thoughts(director_response.content)}"
                    )
                    self.message_manager.add_message(director_response)
                    self.turn_count += 1

                    await self.checkpoint_manager.save_checkpoint_if_needed(
                        self.message_manager, self.turn_count
                    )

        recent_messages = self.message_manager.get_recent_messages(limit=10)
        await self.memory_manager.update_all_memories(recent_messages, self.turn_count)
        print(f"\n🧠 Final memory update completed for all agents")

        return self.message_manager.get_original_messages()
