import random
from typing import List, Optional
from autogen_core.models import LLMMessage, UserMessage
from autogen_agentchat.messages import TextMessage
from agent import DirectorAgent, ParticipantAgent
from utils import extract_content_without_thoughts


class DatingShowChat:
    """Chat class for the dating show."""

    def __init__(
        self,
        participants: List[ParticipantAgent],
        director: DirectorAgent,
        max_turns: int = 5,
        director_intervention_probability: float = 0.05,
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
        self.director_intervention_probability = director_intervention_probability
        self.messages: List[LLMMessage] = []  # Original messages with thoughts
        self.sanitized_messages: List[TextMessage] = (
            []
        )  # Sanitized messages for agent consumption
        self.turn_count = 0
        self.participant_speak_counts = {
            p.name: 0 for p in participants
        }  # Track speaking frequency
        self.memory_update_frequency = 3  # Update memory every N turns
        self.save_frequency = 10  # Save checkpoint every N turns
        self.pipeline = pipeline  # Reference to Stage1Pipeline for saving
        self.base_filename = base_filename

    def add_message(self, message: LLMMessage) -> None:
        """
        Add a message to both original and sanitized message lists.

        Args:
            message: The original message to add
        """
        self.messages.append(message)

        # Create sanitized version for agent consumption
        sanitized_message = TextMessage(
            content=extract_content_without_thoughts(message.content),
            source=message.source,
        )
        self.sanitized_messages.append(sanitized_message)

    def get_original_messages(self) -> List[LLMMessage]:
        """Get the original messages with thoughts included."""
        return self.messages.copy()

    def get_sanitized_messages(self) -> List[TextMessage]:
        """Get the sanitized messages without thoughts for agent consumption."""
        return self.sanitized_messages.copy()

    def get_recent_messages(self, limit: int = 8) -> List[TextMessage]:
        """Get the last N sanitized messages for agent consumption."""
        return (
            self.sanitized_messages[-limit:]
            if len(self.sanitized_messages) > limit
            else self.sanitized_messages.copy()
        )

    async def update_agent_memories(self) -> None:
        """Update memory for all agents based on recent conversation."""
        try:
            participant_names = [p.name for p in self.participants]
            recent_messages = self.get_recent_messages(
                limit=10
            )  # Use more messages for memory update

            # Update participant memories
            for participant in self.participants:
                await participant.update_memory(recent_messages, participant_names)

            # Update director memory
            await self.director.update_memory(recent_messages, participant_names)

            print(f"🧠 Updated memories for all agents (turn {self.turn_count})")

        except Exception as e:
            print(f"Error updating agent memories: {e}")

    async def save_checkpoint_if_needed(self) -> None:
        """Save conversation checkpoint if pipeline is available and it's time to save."""
        if (
            self.pipeline
            and self.turn_count > 0
            and self.turn_count % self.save_frequency == 0
        ):
            try:
                # Convert current conversation to structured data
                from stage1_agent_setup import (
                    Stage1Pipeline,
                )  # Import here to avoid circular import

                # Create temporary conversation data for checkpoint
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

                # Convert messages to structured format
                conversation_messages = []
                for msg in self.messages:
                    from utils import extract_content_without_thoughts
                    from models import MessageType

                    # Determine message type based on source
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

                # Create ConversationData structure
                from models import (
                    ConversationData,
                    ParticipantInfo,
                    DirectorInfo,
                    ConversationMessage,
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
                    total_turns=self.turn_count,
                    metadata={
                        "pipeline_stage": 1,
                        "generation_method": "autogen_agents",
                        "checkpoint": True,
                    },
                )

                # Save checkpoint
                self.pipeline.save_checkpoint(conversation_data, self.base_filename)

            except Exception as e:
                print(f"Error saving checkpoint at turn {self.turn_count}: {e}")

    async def _check_director_intervention(self) -> bool:
        """
        Check if the director should intervene based on conversation analysis and probability.

        Returns:
            True if director should intervene, False otherwise
        """
        # Use a combination of director's analysis and probability
        try:
            # Get recent messages for analysis instead of full history
            recent_messages = self.get_recent_messages(limit=8)

            # First check if director thinks intervention is needed based on conversation flow
            director_analysis = await self.director.should_intervene(recent_messages)

            # Then apply probability factor
            random_chance = random.random() < self.director_intervention_probability

            # Combine both factors - director's analysis OR random chance (but weighted towards analysis)
            should_intervene = director_analysis or (
                random_chance and len(recent_messages) > 3
            )

            return should_intervene
        except Exception as e:
            print(f"Error checking director intervention: {e}")
            # Fallback to simple probability-based decision
            return random.random() < self.director_intervention_probability

    async def _select_next_speaker(self) -> Optional[ParticipantAgent]:
        """
        Select the next speaker based on interest scores and organic conversation flow.

        Returns:
            The participant who should speak next, or None if no one wants to speak
        """
        if len(self.participants) == 0:
            return None

        try:
            # Get recent messages for interest scoring instead of full history
            recent_messages = self.get_recent_messages(limit=8)

            # Get interest scores from all participants
            interest_scores = {}
            for participant in self.participants:
                score = await participant.get_speaking_interest(recent_messages)
                interest_scores[participant.name] = score
                print(f"  {participant.name} interest: {score:.2f}")

            # Apply fairness adjustment - boost scores for participants who haven't spoken much
            adjusted_scores = {}
            total_speaks = sum(self.participant_speak_counts.values())

            for participant in self.participants:
                base_score = interest_scores[participant.name]
                speak_count = self.participant_speak_counts[participant.name]

                # Fairness boost: participants who have spoken less get a small boost
                if total_speaks > 0:
                    fairness_boost = 0.1 * (
                        1 - speak_count / max(1, total_speaks / len(self.participants))
                    )
                else:
                    fairness_boost = 0.0

                adjusted_scores[participant.name] = base_score + fairness_boost

            # Find participants with significant interest (above 0.4 threshold)
            interested_participants = [
                p for p in self.participants if adjusted_scores[p.name] > 0.4
            ]

            if not interested_participants:
                # If no one is very interested, lower the threshold
                interested_participants = [
                    p for p in self.participants if adjusted_scores[p.name] > 0.2
                ]

            if not interested_participants:
                # If still no one, just pick randomly
                return random.choice(self.participants)

            # Weight selection by adjusted interest scores
            adjusted_weights = [
                adjusted_scores[p.name] for p in interested_participants
            ]
            selected = random.choices(
                interested_participants, weights=adjusted_weights
            )[0]

            return selected

        except Exception as e:
            print(f"Error selecting next speaker: {e}")
            # Fallback to random selection
            return random.choice(self.participants)

    async def run(self) -> List[LLMMessage]:
        """Run the conversation."""
        print(f"=== Starting Dating Show Conversation ===")

        # Add initial task message
        initial_message = UserMessage(
            content="We are producing a reality dating show. The video should be around 5 minutes and have a dramatic and engaging tone. The goal is to get the participants to fall in love with each other. Director, please start the show. You can also make up an event or a new challenge if you think it's necessary.",
            source="producer",
        )
        self.add_message(initial_message)

        # Director starts the show
        director_intro = self.director.get_director_intro()
        print(f"Director: {director_intro}")
        self.add_message(director_intro)
        self.turn_count += 1

        # Initial memory update after director intro
        await self.update_agent_memories()

        while self.turn_count < self.max_turns:
            print("\n" + "=" * 50)
            print("🎭 Selecting next speaker based on interest...")

            # Select who wants to speak next based on interest
            next_speaker = await self._select_next_speaker()

            if not next_speaker:
                print("No participants want to speak. Conversation ending.")
                break

            print(f"🎤 {next_speaker.name} wants to speak!")
            print("\n" + "=" * 50)
            print(f"Participant {next_speaker.name}'s turn to speak \n")

            # Get response from selected participant using recent messages only
            recent_messages = self.get_recent_messages(limit=8)
            participant_response = await next_speaker._get_participant_response(
                recent_messages
            )
            response_content = extract_content_without_thoughts(
                participant_response.chat_message.content
            )
            print(f"Participant {next_speaker.name}: {response_content}")

            # Debug: Check if response looks like a number instead of conversation
            if response_content.strip().replace(".", "").isdigit():
                print(
                    f"⚠️  WARNING: {next_speaker.name} gave a numeric response instead of conversation: '{response_content}'"
                )

            # Update tracking
            self.add_message(participant_response.chat_message)
            self.participant_speak_counts[next_speaker.name] += 1
            self.turn_count += 1

            # Update agent memories periodically
            if self.turn_count % self.memory_update_frequency == 0:
                await self.update_agent_memories()

            # Save checkpoint periodically
            await self.save_checkpoint_if_needed()

            if self.turn_count >= self.max_turns:
                break

            # Director analyzes conversation and decides whether to intervene
            should_intervene = await self._check_director_intervention()
            if should_intervene:
                print("\n" + "=" * 50)
                print("Director intervention triggered!")
                recent_messages = self.get_recent_messages(limit=8)
                director_response = await self.director.get_director_intervention(
                    recent_messages
                )
                if director_response:
                    print(
                        f"Director: {extract_content_without_thoughts(director_response.content)}"
                    )
                    self.add_message(director_response)
                    self.turn_count += 1

                    # Save checkpoint after director intervention too
                    await self.save_checkpoint_if_needed()

        # Final memory update at the end
        await self.update_agent_memories()
        print(f"\n🧠 Final memory update completed for all agents")

        return self.messages
