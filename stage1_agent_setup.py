"""
Stage 1: Agent Setup and Conversation Generation
This module handles the creation of participants and directors for the dating show.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Tuple, Union
from datetime import datetime
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from agent import AgentFactory, ParticipantAgent, DirectorAgent
from dating_show_chat import DatingShowChat
from models import ConversationData, ParticipantInfo, DirectorInfo, ConversationMessage, MessageType
from utils import extract_content_without_thoughts

load_dotenv(override=True)


class Stage1Pipeline:
    """Pipeline for Stage 1: Agent setup and conversation generation."""
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _create_model_client(self, model_client_config: dict) -> Union[OllamaChatCompletionClient, OpenAIChatCompletionClient]:
        """Create and return model client"""
        if model_client_config.get("client") == "ollama":
            return OllamaChatCompletionClient(model=model_client_config.get("model"))
        elif model_client_config.get("client") == "openai":
            return OpenAIChatCompletionClient(model=model_client_config.get("model"))
        else:
            raise ValueError(f"Invalid client: {model_client_config.get('client')}")
    
    def create_participants_and_director(self, participant_configs: List[dict], director_config: dict) -> Tuple[List[ParticipantAgent], DirectorAgent, List[OllamaChatCompletionClient]]:
        """Create participants and director with custom configurations."""
        agent_factory = AgentFactory()
        participants = []
        
        # Create participants based on configs
        for i, config in enumerate(participant_configs):
            model_client = self._create_model_client(config.get("model_client"))
            participant = agent_factory.create_participant(
                name=config.get("name", f"Participant{i+1}"),
                age=config.get("age", 25),
                personality_traits=config.get("personality_traits", ["friendly"]),
                model_client=model_client
            )
            participants.append(participant)
        
        # Create director
        director = agent_factory.create_director(
            name=director_config.get("name", "Director"),
            model_client=self._create_model_client(director_config.get("model_client"))
        )
        
        return participants, director
    
    def _convert_messages_to_conversation_data(self, messages, participants: List[ParticipantAgent], director: DirectorAgent, max_turns: int, total_turns: int) -> ConversationData:
        """Convert raw messages to ConversationData model."""
        # Convert participants to ParticipantInfo
        participant_infos = [
            ParticipantInfo(
                name=p.name,
                age=p.age,
                personality_traits=p.personality_traits,
                system_message=p.get_system_message()
            ) for p in participants
        ]
        
        # Convert director to DirectorInfo
        director_info = DirectorInfo(
            name=director.name,
            system_message=director.get_system_message()
        )
        
        # Convert messages to ConversationMessage
        conversation_messages = []
        for msg in messages:
            # Determine message type based on source
            if msg.source == "producer":
                msg_type = MessageType.USER
            elif msg.source == "director":
                msg_type = MessageType.DIRECTOR
            elif msg.source in [p.name for p in participants]:
                msg_type = MessageType.PARTICIPANT
            else:
                msg_type = MessageType.ASSISTANT
            
            conversation_msg = ConversationMessage(
                content=extract_content_without_thoughts(msg.content),
                source=msg.source,
                message_type=msg_type
            )
            conversation_messages.append(conversation_msg)
        
        return ConversationData(
            participants=participant_infos,
            director=director_info,
            messages=conversation_messages,
            max_turns=max_turns,
            total_turns=total_turns,
            metadata={
                "pipeline_stage": 1,
                "generation_method": "autogen_agents"
            }
        )
    
    async def run_conversation(self, participants: List[ParticipantAgent], director: DirectorAgent, max_turns: int = 5) -> ConversationData:
        """Run the dating show conversation and return structured data."""
        dating_chat = DatingShowChat(participants=participants, director=director, max_turns=max_turns)
        
        print("=== Dating Show Conversation ===\n")
        messages = await dating_chat.run()
        
        # Convert to structured data
        conversation_data = self._convert_messages_to_conversation_data(
            messages, participants, director, max_turns, dating_chat.turn_count
        )
        
        return conversation_data
    
    def save_conversation(self, conversation_data: ConversationData, filename: str = None) -> str:
        """Save conversation data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Conversation saved to: {filepath}")
        return str(filepath)
    
    async def run_full_stage1(self, max_turns: int = 5, participants_config: dict = None) -> str:
        """Run the complete Stage 1 pipeline."""
        try:
            # Create agents
            if participants_config == None:
                raise ValueError("participants_config is required")

            participants, director = self.create_participants_and_director(
                participants_config.get("participants", []),
                participants_config.get("director", {})
            )
            model_clients = [p.model_client for p in participants] + [director.model_client]
            
            # Run conversation
            conversation_data = await self.run_conversation(participants, director, max_turns)
            
            # Save conversation
            filepath = self.save_conversation(conversation_data)
            
            # Cleanup model clients
            for client in model_clients:
                await client.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error in Stage 1 pipeline: {e}")
            # Cleanup on error
            try:
                for client in model_clients:
                    await client.close()
            except:
                pass
            raise


async def main():
    """Main function for running Stage 1 independently."""
    pipeline = Stage1Pipeline()
    
    # You can customize the configuration here
    participants_config = Path("participants_config.json")
    
    try:
        filepath = await pipeline.run_full_stage1(max_turns=5, participants_config=participants_config)
        print(f"\nStage 1 completed successfully!")
        print(f"Conversation file: {filepath}")
        print(f"\nTo proceed to Stage 2, run:")
        print(f"python stage2_storyboard.py --conversation {filepath}")
        
    except Exception as e:
        print(f"Stage 1 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
