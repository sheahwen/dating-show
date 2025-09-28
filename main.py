import asyncio
import json
from datetime import datetime
from pathlib import Path
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from agent import AgentFactory
from dating_show_chat import DatingShowChat
from conversation_utils import save_conversation_to_file

load_dotenv(override=True)

# p1_model_client = OllamaChatCompletionClient(model="deepseek-r1")
p1_model_client = OllamaChatCompletionClient(model="llama3.2:1b")
p2_model_client = OllamaChatCompletionClient(model="llama3.2")
director_model_client = OllamaChatCompletionClient(model="deepseek-r1")
# director_model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash")


async def main() -> None:
    agent_factory = AgentFactory()
    
    # Create two participants with different personalities
    participant1 = agent_factory.create_participant(
        name="Alex",
        age=28,
        personality_traits=["confident", "adventurous", "outgoing", "funny", "traveler"],
        model_client=p1_model_client
    )
    participant2 = agent_factory.create_participant(
        name="Sarah",
        age=25,
        personality_traits=["shy", "introverted", "artistic", "romantic", "bookworm"],
        model_client=p2_model_client
    )

    participants = [participant1, participant2]
    director = agent_factory.create_director(name="Director", model_client=director_model_client)
    dating_chat = DatingShowChat(participants=participants, director=director, max_turns=4)
    
    print("=== Dating Show Conversation ===\n")

    # Generate conversation
    messages = await dating_chat.run()
    
    # Save conversation to file
    episode_info = {
        "title": "The First Meeting",
        "season": 1,
        "episode": 1,
        "description": "Alex and Sarah meet for the first time at the villa",
        "generated_at": datetime.now().isoformat(),
        "total_messages": len(messages)
    }
    
    conversation_file = save_conversation_to_file(messages, participants, director, episode_info)
    
    print(f"\n✅ Episode generated successfully!")
    print(f"📁 Conversation file: {conversation_file}")
    print(f"📊 Total messages: {len(messages)}")
    print(f"💬 Participant contributions:")
    for participant in participants:
        count = sum(1 for msg in messages if msg.source == participant.name)
        print(f"   {participant.name}: {count} messages")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review the conversation in: {conversation_file}")
    print(f"   2. Run video generation: python generate_video_from_conversation.py --conversation {conversation_file}")
    print(f"   3. Or load it programmatically for video generation")

    await p1_model_client.close()
    await p2_model_client.close()
    await director_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
