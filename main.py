import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from agent import AgentFactory
from dating_show_chat import DatingShowChat

load_dotenv(override=True)

# p1_model_client = OllamaChatCompletionClient(model="deepseek-r1")
p1_model_client = OllamaChatCompletionClient(model="llama3.2:1b")
p2_model_client = OllamaChatCompletionClient(model="llama3.2")
director_model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash")

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
    dating_chat = DatingShowChat(participants=participants, director=director)
    
    print("=== Dating Show Conversation ===\n")

    result = await dating_chat.run()

    # print("\n=== Conversation Summary ===")

    # print(result[2])

    await p1_model_client.close()
    await p2_model_client.close()
    await director_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
