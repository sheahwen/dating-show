#!/usr/bin/env python3
"""
Test script demonstrating the full conversation-to-video workflow.
"""

import asyncio
import json
from pathlib import Path
from autogen_core.models import UserMessage, AssistantMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from main import save_conversation_to_file
from generate_video_from_conversation import generate_video_from_file

load_dotenv(override=True)

def create_sample_conversation():
    """Create a sample conversation for testing."""
    
    # Sample messages
    messages = [
        UserMessage(content="Welcome to our dating show!", source="producer"),
        AssistantMessage(content="Good evening everyone! Welcome to 'Love Villa' - where hearts meet and sparks fly! Tonight, we have two incredible individuals ready to find their perfect match.", source="Director"),
        AssistantMessage(content="Hi everyone! I'm Alex, I'm 28 and I absolutely love adventure. I've traveled to over 20 countries and I'm always looking for the next thrill. I work as a travel photographer, so I get to combine my passion for exploring with my career. I'm here because I believe love is the greatest adventure of all!", source="Alex"),
        AssistantMessage(content="Hello! I'm Sarah, I'm 25, and I'm quite different from Alex I think. I'm more of a homebody - I love painting, reading poetry, and spending quiet evenings with a good book and a cup of tea. I run a small art studio where I teach children to paint. I'm here because I want to find someone who appreciates the beauty in simple, quiet moments.", source="Sarah"),
        AssistantMessage(content="What an interesting contrast! Alex, what draws you to someone like Sarah who seems to appreciate a quieter lifestyle?", source="Director"),
        AssistantMessage(content="You know, that's exactly what intrigues me about Sarah. In all my travels, the most beautiful moments have actually been the quiet ones - watching a sunrise alone on a mountain, or sitting by a lake just thinking. I think Sarah could show me how to find that same peace at home. Plus, there's something really attractive about someone who's passionate about creating art and nurturing others.", source="Alex"),
        AssistantMessage(content="That's... that's really sweet, Alex. I have to admit, your stories about traveling sound amazing. I've always been too scared to venture far from home, but hearing you talk about it makes me think maybe I've been missing out. Maybe you could show me that the world isn't as scary as I think it is?", source="Sarah"),
        AssistantMessage(content="I love seeing this connection already! Sarah, what would be your ideal first date with Alex?", source="Director"),
        AssistantMessage(content="Oh my... well, maybe we could start small? Perhaps Alex could show me how to take photographs around the city - seeing familiar places through his adventurous eyes? And then we could come back to my studio and I could paint what we discovered together. A perfect blend of his world and mine.", source="Sarah"),
        AssistantMessage(content="Sarah, that sounds absolutely perfect! I would love to see the city through your artist's perspective while showing you mine. And painting together sounds incredibly romantic. You know what? I think this could be the start of something really special.", source="Alex"),
    ]
    
    # Sample participant data
    participants_data = [
        {
            "name": "Alex",
            "age": 28,
            "personality_traits": ["confident", "adventurous", "outgoing", "funny", "traveler"],
            "agent_type": "participant",
            "system_message": "You are Alex, a 28-year-old participant on a dating show..."
        },
        {
            "name": "Sarah",
            "age": 25,
            "personality_traits": ["shy", "introverted", "artistic", "romantic", "bookworm"],
            "agent_type": "participant",
            "system_message": "You are Sarah, a 25-year-old participant on a dating show..."
        }
    ]
    
    director_data = {
        "name": "Director",
        "agent_type": "director",
        "system_message": "You are the director of this reality dating show..."
    }
    
    # Create conversation data structure
    episode_info = {
        "title": "First Sparks",
        "season": 1,
        "episode": 1,
        "description": "Alex and Sarah meet for the first time and discover an unexpected connection between adventure and art",
        "generated_at": "2024-01-15T20:30:00",
        "total_messages": len(messages)
    }
    
    conversation_data = {
        "episode_info": episode_info,
        "participants": participants_data,
        "director": director_data,
        "conversation": [
            {
                "speaker": msg.source,
                "content": msg.content,
                "clean_content": msg.content,  # Same since no <think> tags in sample
                "message_type": type(msg).__name__
            }
            for msg in messages
        ],
        "metadata": {
            "participant_speak_counts": {
                "Alex": sum(1 for msg in messages if msg.source == "Alex"),
                "Sarah": sum(1 for msg in messages if msg.source == "Sarah"),
                "Director": sum(1 for msg in messages if msg.source == "Director")
            },
            "director_interventions": sum(1 for msg in messages if msg.source == "Director"),
            "total_duration_estimate": len(messages) * 15
        }
    }
    
    # Save to file
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)
    
    filename = "sample_episode_first_sparks.json"
    filepath = conversations_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sample conversation created: {filepath}")
    return str(filepath)

async def test_workflow():
    """Test the complete workflow from conversation to video."""
    
    print("🧪 TESTING FULL CONVERSATION-TO-VIDEO WORKFLOW")
    print("="*60)
    
    # Step 1: Create sample conversation
    print("\n📝 Step 1: Creating sample conversation...")
    conversation_file = create_sample_conversation()
    
    # Step 2: Generate video from conversation
    print(f"\n🎬 Step 2: Generating video from conversation...")
    print(f"📁 Using conversation file: {conversation_file}")
    
    try:
        results = await generate_video_from_file(conversation_file, video_provider="mock")
        
        if results:
            print(f"\n✅ WORKFLOW TEST SUCCESSFUL!")
            print(f"📊 Results summary:")
            print(f"   Episode: {results['episode_info']['title']}")
            print(f"   Segments: {results['total_segments']}")
            print(f"   Duration: {results['estimated_duration']} seconds")
            print(f"   Storyboard: {results['storyboard_file']}")
            
            # Show what was generated
            print(f"\n📁 Generated files:")
            print(f"   Conversation: {conversation_file}")
            print(f"   Storyboard: {results['storyboard_file']}")
            
            storyboard_dir = Path("video_prompts")
            if storyboard_dir.exists():
                prompt_files = list(storyboard_dir.glob("*.json"))
                if prompt_files:
                    print(f"   Video prompts: {len(prompt_files)} files in video_prompts/")
            
            video_dir = Path("generated_videos") 
            if video_dir.exists():
                video_files = list(video_dir.glob("*"))
                if video_files:
                    print(f"   Mock videos: {len(video_files)} files in generated_videos/")
            
            print(f"\n🎯 Next steps:")
            print(f"   1. Review the storyboard: {results['storyboard_file']}")
            print(f"   2. Check video prompts in: video_prompts/")
            print(f"   3. To generate real videos, use: --provider runway or --provider pika")
            print(f"   4. Manage conversations with: python conversation_manager.py list")
            
        else:
            print(f"❌ WORKFLOW TEST FAILED")
            
    except Exception as e:
        print(f"❌ Error during workflow test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function."""
    await test_workflow()

if __name__ == "__main__":
    asyncio.run(main())
