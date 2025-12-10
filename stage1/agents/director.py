from autogen_agentchat.messages import LLMMessage, TextMessage
from autogen_core.models import AssistantMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

from stage1.agents.base import AgentType, BaseAgent
from stage1.prompts import (
    DIRECTOR_INTRO_MESSAGE,
    DIRECTOR_SYSTEM_MESSAGE,
    build_director_analysis_prompt,
    build_director_intervention_prompt,
)


class DirectorAgent(BaseAgent):
    """Agent representing the director of the dating show."""

    def __init__(self, name: str, model_client: OllamaChatCompletionClient):
        super().__init__(name, AgentType.DIRECTOR, model_client)

    def get_system_message(self) -> str:
        return DIRECTOR_SYSTEM_MESSAGE.format(name=self.name)

    def get_details(self) -> dict:
        """Get detailed information about the director."""
        return {
            "name": self.name,
            "show_format": getattr(self, "show_format", None),
            "directing_style": getattr(self, "directing_style", None),
            "agent_type": self.agent_type.value,
            "system_message": self.get_system_message(),
        }

    def get_director_intro(self) -> AssistantMessage:
        """Get the director's introduction."""
        return AssistantMessage(content=DIRECTOR_INTRO_MESSAGE, source="director")

    async def should_intervene(self, recent_messages: list[TextMessage]) -> bool:
        """Decide if intervention is needed."""
        if len(recent_messages) < 3:
            return False

        memory_context = self.get_memory_context()
        analysis_prompt = build_director_analysis_prompt(memory_context)

        try:
            analysis_messages = recent_messages + [
                TextMessage(content=analysis_prompt, source="producer")
            ]
            response = await self.agent.on_messages(
                analysis_messages, cancellation_token=None
            )
            decision = response.chat_message.content.strip().upper()
            return "INTERVENE" in decision
        except Exception as e:
            print(f"Error in director analysis: {e}")
            return False

    async def get_director_intervention(
        self, recent_messages: list[TextMessage]
    ) -> LLMMessage | None:
        """Get director intervention response based on memory and recent conversation context."""
        memory_context = self.get_memory_context()
        intervention_prompt = build_director_intervention_prompt(memory_context)

        try:
            context_messages = recent_messages + [
                TextMessage(content=intervention_prompt, source="producer")
            ]
            response = await self.agent.on_messages(
                context_messages, cancellation_token=None
            )
            return response.chat_message
        except Exception as e:
            print(f"Error getting director intervention: {e}")
            return None
