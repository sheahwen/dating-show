from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import LLMMessage, TextMessage, UserMessage
from utils import extract_content_without_thoughts
from autogen_core.models import AssistantMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient


class AgentType(Enum):
    PARTICIPANT = "participant"
    DIRECTOR = "director"

class BaseAgent(ABC):
    """Base class for all dating show agents."""

    def __init__(self, name: str, agent_type: AgentType, model_client: OllamaChatCompletionClient):
        self.name = name
        self.agent_type = agent_type
        self.model_client = model_client
        self._agent = self._create_agent()

    @abstractmethod
    def get_system_message(self) -> str:
        """Get the system message for the agent."""
        pass

    def _create_agent(self) -> AssistantAgent:
        """Create the underlying AutoGen agent."""
        print(f'Creating agent {self.name} of type {self.agent_type} with system message {self.get_system_message()}')
        return AssistantAgent(name=self.name, model_client=self.model_client, system_message=self.get_system_message(), reflect_on_tool_use=True,model_client_stream=False)

    @property
    def agent(self) -> AssistantAgent:
        """Access to the underlying AutoGen agent."""
        return self._agent

class ParticipantAgent(BaseAgent):
    """Agent representing a participant in the dating show."""

    def __init__(self, name: str, model_client: OllamaChatCompletionClient, personality_traits: list[str] = None, age: int = 25):
        self.personality_traits = personality_traits
        self.age = age
        self.model_client = model_client
        super().__init__(name, AgentType.PARTICIPANT, model_client)

    def get_system_message(self) -> str:
        traits_str = ", ".join(self.personality_traits) if self.personality_traits else "boring and extremely logical person"
        return f"""You are {self.name}, a {self.age}-year-old participant on a dating show. Your personality is {traits_str}. You're here to find love and make genuine connections. Be authentic, engaging, and true to your personality while participating in challenges and conversations."""

    def get_details(self) -> dict:
        """Get detailed information about the participant."""
        return {
            "name": self.name,
            "age": self.age,
            "personality_traits": self.personality_traits,
            "agent_type": self.agent_type.value,
            "system_message": self.get_system_message()
        }

    async def get_speaking_interest(self, messages: List[TextMessage]) -> float:
        """
        Evaluate how much this participant wants to speak based on the current conversation.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            Interest score from 0.0 (no interest) to 1.0 (very interested)
        """
        try:
            interest_prompt = """Based on the current conversation, rate how much you want to respond or contribute right now.
            Consider:
            - Are you being directly addressed or mentioned?
            - Is the topic something you're passionate about or can relate to?
            - Do you have something meaningful to add?
            - Are you feeling engaged with the conversation flow?
            - Does someone need a response or clarification?
            
            Respond with only a number from 0.0 to 1.0:
            - 0.0 = No interest in speaking right now
            - 0.3 = Mild interest, could contribute if needed
            - 0.5 = Moderate interest, have something to say
            - 0.7 = High interest, want to respond
            - 1.0 = Very high interest, must respond/react
            
            Just respond with the number, nothing else."""
            
            interest_messages = messages + [TextMessage(content=interest_prompt, source="system")]
            response = await self.agent.on_messages(interest_messages, cancellation_token=None)
            
            # Extract the numeric score
            content = response.chat_message.content.strip()
            try:
                # Try to extract a float from the response
                score = float(content.split()[0])  # Get first number in case there's extra text
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except (ValueError, IndexError):
                # If parsing fails, return a default medium interest
                return 0.3
                
        except Exception as e:
            print(f"Error getting speaking interest for {self.name}: {e}")
            return 0.3  # Default medium interest

    async def _get_participant_response(self, messages: List[TextMessage]) -> Optional[LLMMessage]:
        """Get response from a participant agent."""
        try:
            # Messages are already sanitized, so we can use them directly
            response = await self.agent.on_messages(messages, cancellation_token=None)

            return response
        except Exception as e:
            print(f"Error getting response from {self.name}: {e}")
            return None
        

class DirectorAgent(BaseAgent):
    """Agent representing the director of the dating show."""

    def __init__(self, name: str, model_client: OllamaChatCompletionClient):
        self.show_format = "reality_dating_show"
        self.directing_style = "dramatic_engaging"
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
            "system_message": self.get_system_message()
        }

    def get_director_intro(self) -> str:
        """Get the director's introduction."""
        content = "You’re all meeting at the villa for the first time. I hope you all will spend the the next few days having fun. Remember: everyone here is just as excited and maybe a little nervous as you are. Go ahead and meet each other."
        return AssistantMessage(content=content, source="director")

    async def should_intervene(self, conversation_messages: List[TextMessage]) -> bool:
        """
        Analyze the conversation and decide if director intervention is needed.
        
        Args:
            conversation_messages: Recent conversation messages to analyze
            
        Returns:
            True if intervention is needed, False otherwise
        """
        if len(conversation_messages) < 6:  # Need at least a few messages to analyze
            return False
            
        # Create analysis prompt
        analysis_prompt = """Analyze the recent conversation between the dating show participants. 
        Determine if you, as the director, should intervene based on the following criteria:
        
        1. Is the conversation getting stale or repetitive?
        2. Are the participants not connecting well?
        3. Is there tension that needs to be addressed?
        4. Would a new topic, challenge, or question help move things forward?
        5. Has the conversation been going on too long without direction?
        
        Respond with only "INTERVENE" if you should step in, or "CONTINUE" if the conversation should flow naturally.
        Consider that interventions should be meaningful and add value to the show."""
        
        try:
            # Add the analysis prompt as a message to get director's decision
            analysis_messages = conversation_messages + [TextMessage(content=analysis_prompt, source="producer")]
            response = await self.agent.on_messages(analysis_messages, cancellation_token=None)
            
            # Check if the response indicates intervention is needed
            decision = response.chat_message.content.strip().upper()
            return "INTERVENE" in decision
        except Exception as e:
            print(f"Error in director analysis: {e}")
            return False

    async def get_director_intervention(self, conversation_messages: List[TextMessage]) -> LLMMessage:
        """
        Get director intervention response based on the conversation context.
        
        Args:
            conversation_messages: Recent conversation messages for context
            
        Returns:
            Director's intervention message
        """
        intervention_prompt = """Based on the current conversation, intervene meaningfully. You can:
            1. Introduce a new topic or challenge
            2. Ask a specific question to one or both participants  
            3. Make a comment on what you've observed
            4. Change the direction of the conversation
            5. Create a dramatic moment or revelation

            Make your intervention engaging and meaningful for the dating show. Be specific and relevant to what just happened."""
        
        try:
            # Provide conversation context for the intervention
            context_messages = conversation_messages + [TextMessage(content=intervention_prompt, source="producer")]
            response = await self.agent.on_messages(context_messages, cancellation_token=None)
            return response.chat_message
        except Exception as e:
            print(f"Error getting director intervention: {e}")
            return None


class AgentFactory:
    """Factory class for create and manage all agents."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def create_participant(self, name:str, model_client: OllamaChatCompletionClient, personality_traits: list[str] = None, age: int = 25, ) -> ParticipantAgent:
        """Create a new participant agent."""
        if name in self.agents:
            raise ValueError(f"Agent with name {name} already exists.")
        agent = ParticipantAgent(name, model_client, personality_traits, age)
        self.agents[name] = agent
        return agent

    def create_director(self, name: str, model_client: OllamaChatCompletionClient) -> DirectorAgent:
        """Create a new director agent."""
        print(self)
        if name in self.agents:
            raise ValueError(f"Agent with name {name} already exists.")
        agent = DirectorAgent(name, model_client)
        self.agents[name] = agent
        return agent

    def get_agent(self, name:str) -> BaseAgent:
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