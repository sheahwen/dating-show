from typing import Optional

from autogen_agentchat.messages import LLMMessage, TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

from stage1.agents.base import AgentType, BaseAgent
from stage1.prompts import (
    CONVERSATION_PROMPT,
    PARTICIPANT_SYSTEM_MESSAGE,
    build_interest_prompt,
)


class ParticipantAgent(BaseAgent):
    """Agent representing a participant in the dating show."""

    def __init__(
        self,
        name: str,
        model_client: OllamaChatCompletionClient,
        personality_traits: list[str] | None = None,
        age: int = 25,
    ):
        self.personality_traits = personality_traits
        self.age = age
        self.model_client = model_client
        super().__init__(name, AgentType.PARTICIPANT, model_client)

    def get_system_message(self) -> str:
        traits_str = (
            ", ".join(self.personality_traits)
            if self.personality_traits
            else "boring and extremely logical person"
        )
        return PARTICIPANT_SYSTEM_MESSAGE.format(
            name=self.name, age=self.age, traits_str=traits_str
        )

    def get_details(self) -> dict:
        """Get detailed information about the participant."""
        return {
            "name": self.name,
            "age": self.age,
            "personality_traits": self.personality_traits,
            "agent_type": self.agent_type.value,
            "system_message": self.get_system_message(),
        }

    async def get_speaking_interest(self, recent_messages: list[TextMessage]) -> float:
        """
        Evaluate how much this participant wants to speak based on recent conversation and memory.
        """
        try:
            memory_context = self.get_memory_context()
            interest_prompt = build_interest_prompt(memory_context)
            interest_messages = recent_messages + [
                TextMessage(content=interest_prompt, source="system")
            ]
            response = await self.agent.on_messages(
                interest_messages, cancellation_token=None
            )
            content = response.chat_message.content.strip()
            try:
                score = float(content.split()[0])
                return max(0.0, min(1.0, score))
            except (ValueError, IndexError):
                print(
                    f"⚠️  ALERT: Failed to parse interest score for {self.name}. Content: '{content}'. Using default score 0.3"
                )
                return 0.3

        except Exception as e:
            print(f"Error getting speaking interest for {self.name}: {e}")
            return 0.3

    async def _get_participant_response(
        self, recent_messages: list[TextMessage]
    ) -> Optional[LLMMessage]:
        """Get response from a participant agent using memory and recent messages."""
        try:
            system_msg = TextMessage(content=self.get_system_message(), source="system")
            memory_context = TextMessage(
                content=self.get_memory_context(), source="system"
            )
            conversation_prompt = TextMessage(
                content=CONVERSATION_PROMPT, source="system"
            )

            conversation_messages = (
                [system_msg, memory_context] + recent_messages + [conversation_prompt]
            )  # type: ignore

            response = await self.agent.on_messages(
                conversation_messages, cancellation_token=None
            )
            return response
        except Exception as e:
            print(f"Error getting response from {self.name}: {e}")
            return None
