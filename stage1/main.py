"""
Stage 1: Agent Setup and Conversation Generation
This module handles the creation of participants and directors for the dating show.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

from dotenv import load_dotenv

from common.clients import create_model_client
from common.io import timestamped_filename
from common.models import ConversationData
from stage1 import AgentFactory, DirectorAgent, ParticipantAgent
from stage1.chat import DatingShowChat
from stage1.helpers import (
    convert_messages_to_conversation_data,
    generate_base_filename,
    save_checkpoint,
    save_conversation,
)

load_dotenv(override=True)


class Stage1Pipeline:
    """Pipeline for Stage 1: Agent setup and conversation generation."""

    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_frequency = 10  # Save every N turns (passed into chat)
        self.checkpoint_counter = 0

    def _create_model_client(self, model_client_config: dict):
        """Create and return a chat model client."""
        return create_model_client(
            model_client_config, default_client="ollama", default_model="llama3.2"
        )

    def create_participants_and_director(
        self, participants_config: List[dict], director_config: dict
    ) -> Tuple[List[ParticipantAgent], DirectorAgent, list]:
        """Create participants and director with custom configurations."""
        agent_factory = AgentFactory()
        participants = []

        # Create participants based on configs
        for i, config in enumerate(participants_config):
            model_client = self._create_model_client(config.get("model_client"))
            participant = agent_factory.create_participant(
                name=config.get("name", f"Participant{i+1}"),
                age=config.get("age", 25),
                personality_traits=config.get(
                    "personality_traits", ["super boring and dull"]
                ),
                model_client=model_client,
            )
            participants.append(participant)

        # Create director
        director = agent_factory.create_director(
            name=director_config.get("name", "Director"),
            model_client=self._create_model_client(director_config.get("model_client")),
        )

        return participants, director

    async def run_conversation(
        self,
        participants: List[ParticipantAgent],
        director: DirectorAgent,
        max_turns: int = 5,
        base_filename: str = None,
    ) -> ConversationData:
        """Run the dating show conversation and return structured data."""
        dating_chat = DatingShowChat(
            participants=participants,
            director=director,
            max_turns=max_turns,
            save_frequency=self.save_frequency,
            pipeline=self,  # Pass pipeline reference for saving
            base_filename=base_filename,
        )

        print("=== Dating Show Conversation ===\n")
        messages = await dating_chat.run()

        # Convert to structured data
        conversation_data = convert_messages_to_conversation_data(
            messages, participants, director, max_turns, dating_chat.turn_count
        )

        return conversation_data

    def save_conversation(
        self, conversation_data: ConversationData, filename: str = None
    ) -> str:
        """Save conversation data to JSON file."""
        filepath = save_conversation(conversation_data, self.output_dir, filename)
        print(f"Conversation saved to: {filepath}")
        return filepath

    def save_checkpoint(
        self,
        conversation_data: ConversationData,
        base_filename: str = None,
        is_final: bool = False,
    ) -> str:
        """Save conversation checkpoint with incremental naming."""
        filepath, self.checkpoint_counter = save_checkpoint(
            conversation_data,
            self.output_dir,
            base_filename,
            self.checkpoint_counter,
            is_final=is_final,
        )
        if is_final:
            print(f"✅ Final conversation saved to: {filepath}")
        else:
            print(f"💾 Checkpoint {self.checkpoint_counter} saved to: {filepath}")
        return filepath

    async def run_full_stage1(
        self,
        max_turns: int = 5,
        participants_config: dict = None,
        director_config: dict = None,
    ) -> str:
        """Run the complete Stage 1 pipeline."""
        try:
            if participants_config == None:
                raise ValueError("participants_config is required")

            if director_config == None:
                raise ValueError("director_config is required")

            # Generate base filename for checkpoints
            base_filename = generate_base_filename("conversation")

            participants, director = self.create_participants_and_director(
                participants_config.get("participants", []),
                director_config.get("director", {}),
            )
            model_clients = [p.model_client for p in participants] + [
                director.model_client
            ]

            # Run conversation with incremental saving
            conversation_data = await self.run_conversation(
                participants, director, max_turns, base_filename
            )

            # Save final conversation
            final_filepath = self.save_checkpoint(
                conversation_data, base_filename, is_final=True
            )

            # Cleanup model clients
            print(f"Closing model clients. Count: {len(model_clients)}")
            for client in model_clients:
                await client.close()

            return final_filepath

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

    participants_config_path = Path("participants_config.json")
    director_config_path = Path("director_config.json")

    participants_config = (
        json.load(open(participants_config_path))
        if participants_config_path.exists()
        else None
    )
    director_config = (
        json.load(open(director_config_path)) if director_config_path.exists() else None
    )

    try:
        filepath = await pipeline.run_full_stage1(
            max_turns=5,
            participants_config=participants_config,
            director_config=director_config,
        )
        print(f"\nStage 1 completed successfully!")
        print(f"Conversation file: {filepath}")
        print(f"\nTo proceed to Stage 2, run:")
        print(f"python stage2_storyboard.py --conversation {filepath}")

    except Exception as e:
        print(f"Stage 1 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
