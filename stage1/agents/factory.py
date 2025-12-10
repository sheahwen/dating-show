from typing import Dict

from autogen_ext.models.ollama import OllamaChatCompletionClient

from stage1.agents.base import AgentType, BaseAgent
from stage1.agents.director import DirectorAgent
from stage1.agents.participant import ParticipantAgent


class AgentFactory:
    """Factory class to create and manage all agents."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def create_participant(
        self,
        name: str,
        model_client: OllamaChatCompletionClient,
        personality_traits: list[str] | None = None,
        age: int = 25,
    ) -> ParticipantAgent:
        """Create a new participant agent."""
        if name in self.agents:
            raise ValueError(f"Agent with name {name} already exists.")
        agent = ParticipantAgent(name, model_client, personality_traits, age)
        self.agents[name] = agent
        return agent

    def create_director(
        self, name: str, model_client: OllamaChatCompletionClient
    ) -> DirectorAgent:
        """Create a new director agent."""
        if name in self.agents:
            raise ValueError(f"Agent with name {name} already exists.")
        agent = DirectorAgent(name, model_client)
        self.agents[name] = agent
        return agent

    def get_agent(self, name: str) -> BaseAgent:
        """Get an agent by name."""
        if name not in self.agents:
            raise ValueError(f"Agent with name {name} does not exist.")
        return self.agents[name]

    def get_agents_by_type(self, agent_type: AgentType) -> list[BaseAgent]:
        """Get all agents of a specific type."""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]

    def get_all_agents(self) -> list[BaseAgent]:
        """Get all created agents."""
        return list(self.agents.values())

