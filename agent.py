from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import LLMMessage, TextMessage, UserMessage
from autogen_core.models import AssistantMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient


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
        pass

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
        self, recent_messages: List[TextMessage], participant_names: List[str]
    ) -> None:
        """
        Update the agent's memory with a summary of recent conversation and participant impressions.

        Args:
            recent_messages: Recent conversation messages to incorporate
            participant_names: Names of all participants for context
        """
        try:
            # Create memory update prompt
            participants_list = ", ".join(participant_names)
            memory_prompt = f"""Update your memory based on the recent conversation. Your memory should be a concise summary (max 300 words) that includes:

1. WHO: List of participants ({participants_list}) and key details about each
2. IMPRESSIONS: Your personal impressions of each participant and their relationships
3. CURRENT SITUATION: What's happening in the show right now
4. KEY EVENTS: Important moments or developments that have occurred

Current memory: {self.memory if self.memory else "None yet - this is the beginning."}

Based on the recent conversation, update your memory to reflect new information while keeping the most important details. Focus on what's most relevant for understanding the current situation and relationships. 

Critical: Do not exceed 300 words.

Respond with only the updated memory summary, nothing else."""

            # Add memory prompt to recent messages
            memory_messages = recent_messages + [
                TextMessage(content=memory_prompt, source="system")
            ]
            response = await self.agent.on_messages(
                memory_messages, cancellation_token=None
            )

            # Update memory with the response
            self.memory = response.chat_message.content.strip()

            # Ensure memory doesn't exceed 300 words
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


class ParticipantAgent(BaseAgent):
    """Agent representing a participant in the dating show."""

    def __init__(
        self,
        name: str,
        model_client: OllamaChatCompletionClient,
        personality_traits: list[str] = None,
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
        return f"""You are {self.name}, a {self.age}-year-old participant on a dating show. Your personality is {traits_str}. You're here to find love and make genuine connections. Be authentic, engaging, and true to your personality while participating in challenges and conversations."""

    def get_details(self) -> dict:
        """Get detailed information about the participant."""
        return {
            "name": self.name,
            "age": self.age,
            "personality_traits": self.personality_traits,
            "agent_type": self.agent_type.value,
            "system_message": self.get_system_message(),
        }

    async def get_speaking_interest(self, recent_messages: List[TextMessage]) -> float:
        """
        Evaluate how much this participant wants to speak based on recent conversation and memory.

        Args:
            recent_messages: Last 8 messages from the conversation

        Returns:
            Interest score from 0.0 (no interest) to 1.0 (very interested)
        """
        try:
            memory_context = self.get_memory_context()

            interest_prompt = f"""INTEREST SCORING TASK: Based on your memory and the recent conversation, rate how much you want to respond or contribute right now.

{memory_context}

Consider:
- Are you being directly addressed or mentioned?
- Is the topic something you're passionate about or can relate to?
- Do you have something meaningful to add based on your personality and experiences?
- Are you feeling engaged with the conversation flow?
- Does someone need a response or clarification?
- Based on your memory, is this a good time for you to speak?

IMPORTANT: This is only for scoring your interest level, NOT for actual conversation.
Respond with ONLY a number from 0.0 to 1.0:
- 0.0 = No interest in speaking right now
- 0.3 = Mild interest, could contribute if needed
- 0.5 = Moderate interest, have something to say
- 0.7 = High interest, want to respond
- 1.0 = Very high interest, must respond/react

Just respond with the number, nothing else. Do not provide conversation content."""

            interest_messages = recent_messages + [
                TextMessage(content=interest_prompt, source="system")
            ]
            response = await self.agent.on_messages(
                interest_messages, cancellation_token=None
            )

            # Extract the numeric score
            content = response.chat_message.content.strip()

            try:
                # Try to extract a float from the response
                score = float(
                    content.split()[0]
                )  # Get first number in case there's extra text
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except (ValueError, IndexError):
                # If parsing fails, return a default medium interest
                print(
                    f"⚠️  ALERT: Failed to parse interest score for {self.name}. Content: '{content}'. Using default score 0.3"
                )
                return 0.3

        except Exception as e:
            print(f"Error getting speaking interest for {self.name}: {e}")
            return 0.3  # Default medium interest

    async def _get_participant_response(
        self, recent_messages: List[TextMessage]
    ) -> Optional[LLMMessage]:
        """Get response from a participant agent using memory and recent messages."""
        try:
            # Include system message at every turn
            system_msg = TextMessage(content=self.get_system_message(), source="system")

            # Add memory context
            memory_context = TextMessage(
                content=self.get_memory_context(), source="system"
            )

            # Conversation prompt to guide natural response
            conversation_prompt = TextMessage(
                content="Now respond naturally as yourself in this dating show conversation. Be authentic, engaging, and true to your personality. This is your chance to connect with the other participants.",
                source="system",
            )

            # Prepare final message list: [system message] + [memory context] + [recent chat history] + [conversation prompt]
            conversation_messages = [system_msg, memory_context] + recent_messages + [conversation_prompt]  # type: ignore

            # Create conversation messages with the prompt
            response = await self.agent.on_messages(
                conversation_messages, cancellation_token=None
            )

            return response
        except Exception as e:
            print(f"Error getting response from {self.name}: {e}")
            return None


class DirectorAgent(BaseAgent):
    """Agent representing the director of the dating show."""

    def __init__(self, name: str, model_client: OllamaChatCompletionClient):
        super().__init__(name, AgentType.DIRECTOR, model_client)

    def get_system_message(self) -> str:
        return f"""You are {self.name}, the director of this reality dating show. Your directing style is dramatic and engaging. You are responsible for:
- Orchestrating challenges and activities for the participants
- Creating dramatic moments and storylines
- Managing the flow and pacing of the show
- Making decisions about eliminations, dates, and special events
- Providing commentary and narration when needed
- Ensuring the show is entertaining and engaging for viewers

You have authority over the show's proceedings and should guide conversations, introduce new elements, and create compelling television moments while maintaining fairness among participants."""

    def get_details(self) -> dict:
        """Get detailed information about the director."""
        return {
            "name": self.name,
            "show_format": self.show_format,
            "directing_style": self.directing_style,
            "agent_type": self.agent_type.value,
            "system_message": self.get_system_message(),
        }

    def get_director_intro(self) -> str:
        """Get the director's introduction."""
        content = "You’re all meeting at the villa for the first time. I hope you all will spend the the next few days having fun. Remember: everyone here is just as excited and maybe a little nervous as you are. Go ahead and meet each other."
        return AssistantMessage(content=content, source="director")

    async def should_intervene(self, recent_messages: List[TextMessage]) -> bool:
        """
        Analyze the recent conversation and memory to decide if director intervention is needed.

        Args:
            recent_messages: Recent conversation messages to analyze

        Returns:
            True if intervention is needed, False otherwise
        """
        if len(recent_messages) < 3:  # Need at least a few messages to analyze
            return False

        memory_context = self.get_memory_context()

        # Create analysis prompt
        analysis_prompt = f"""Analyze the recent conversation between the dating show participants using your memory and recent messages.

{memory_context}

Determine if you, as the director, should intervene based on the following criteria:

1. Is the conversation getting stale or repetitive?
2. Are the participants not connecting well?
3. Is there tension that needs to be addressed?
4. Would a new topic, challenge, or question help move things forward?
5. Has the conversation been going on too long without direction?
6. Based on your memory, is this a good moment for dramatic intervention?

Respond with only "INTERVENE" if you should step in, or "CONTINUE" if the conversation should flow naturally.
Consider that interventions should be meaningful and add value to the show."""

        try:
            # Add the analysis prompt as a message to get director's decision
            analysis_messages = recent_messages + [
                TextMessage(content=analysis_prompt, source="producer")
            ]
            response = await self.agent.on_messages(
                analysis_messages, cancellation_token=None
            )

            # Check if the response indicates intervention is needed
            decision = response.chat_message.content.strip().upper()
            return "INTERVENE" in decision
        except Exception as e:
            print(f"Error in director analysis: {e}")
            return False

    async def get_director_intervention(
        self, recent_messages: List[TextMessage]
    ) -> LLMMessage:
        """
        Get director intervention response based on memory and recent conversation context.

        Args:
            recent_messages: Recent conversation messages for context

        Returns:
            Director's intervention message
        """
        memory_context = self.get_memory_context()

        intervention_prompt = f"""Based on your memory and the current conversation, intervene meaningfully. 

{memory_context}

You can:
1. Introduce a new topic or challenge based on what you know about the participants
2. Ask a specific question to one or both participants  
3. Make a comment on what you've observed from your memory
4. Change the direction of the conversation
5. Create a dramatic moment or revelation
6. Reference past events or relationships from your memory

Make your intervention engaging and meaningful for the dating show. Be specific and relevant to both recent events and your overall memory of the show."""

        try:
            # Provide conversation context for the intervention
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


class AgentFactory:
    """Factory class for create and manage all agents."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def create_participant(
        self,
        name: str,
        model_client: OllamaChatCompletionClient,
        personality_traits: list[str] = None,
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
        return [
            agent for agent in self.agents.values() if agent.agent_type == agent_type
        ]

    def get_all_agents(self) -> list[BaseAgent]:
        """Get all created agents."""
        return list(self.agents.values())
