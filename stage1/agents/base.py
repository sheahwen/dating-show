from abc import ABC, abstractmethod
from enum import Enum

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from stage1.prompts import build_memory_prompt


class AgentType(Enum):
    PARTICIPANT = "participant"
    DIRECTOR = "director"


class BaseAgent(ABC):
    """Base class for all dating show agents."""

    def __init__(
        self, name: str, agent_type: AgentType, model_client: OllamaChatCompletionClient
    ):
        self.name = name
        self.agent_type = agent_type
        self.model_client = model_client
        self._agent = self._create_agent()
        self.memory: str = ""  # Agent's summarized memory (max 300 words)

    @abstractmethod
    def get_system_message(self) -> str:
        """Get the system message for the agent."""
        raise NotImplementedError

    def _create_agent(self) -> AssistantAgent:
        """Create the underlying AutoGen agent."""
        print(
            f"Creating agent {self.name} of type {self.agent_type} with system message {self.get_system_message()}"
        )
        return AssistantAgent(
            name=self.name,
            model_client=self.model_client,
            system_message=self.get_system_message(),
            reflect_on_tool_use=True,
            model_client_stream=False,
        )

    @property
    def agent(self) -> AssistantAgent:
        """Access to the underlying AutoGen agent."""
        return self._agent

    async def update_memory(
        self, recent_messages: list[TextMessage], participant_names: list[str]
    ) -> None:
        """
        Update the agent's memory with a summary of recent conversation and participant impressions.
        """

        try:
            participants_list = ", ".join(participant_names)
            current_memory = (
                self.memory if self.memory else "None yet - this is the beginning."
            )
            memory_prompt = build_memory_prompt(participants_list, current_memory)
            memory_messages = recent_messages + [
                TextMessage(content=memory_prompt, source="system")
            ]
            response = await self.agent.on_messages(
                memory_messages, cancellation_token=None
            )
            self.memory = response.chat_message.content.strip()

            words = self.memory.split()
            if len(words) > 300:
                self.memory = " ".join(words[:300]) + "..."

        except Exception as e:
            print(f"Error updating memory for {self.name}: {e}")

    def get_memory_context(self) -> str:
        """Get the current memory as context for prompts."""
        if not self.memory:
            return "This is the beginning of the show - no prior context yet."
        return f"Your memory of the show so far: {self.memory}"
