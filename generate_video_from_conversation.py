#!/usr/bin/env python3
"""
Generate videos from saved conversations using the three-stage pipeline.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage, AssistantMessage
from dotenv import load_dotenv

from video_pipeline import VideoPipeline
from conversation_utils import load_conversation_from_file

load_dotenv(override=True)

def reconstruct_messages_from_conversation_data(conversation_data: Dict) -> List:
    """Reconstruct message objects from saved conversation data."""
    messages = []
    
    for msg_data in conversation_data['conversation']:
        speaker = msg_data['speaker']
        content = msg_data['content']
        msg_type = msg_data['message_type']
        
        # Reconstruct appropriate message type
        if msg_type == "UserMessage" or speaker == "producer":
            message = UserMessage(content=content, source=speaker)
        else:
            message = AssistantMessage(content=content, source=speaker)
        
        messages.append(message)
    
    return messages

async def generate_video_from_file(conversation_file: str, video_provider: str = "mock"):
    """Generate video from saved conversation file."""
    
    print(f"🎬 GENERATING VIDEO FROM CONVERSATION FILE")
    print(f"📁 File: {conversation_file}")
    print(f"🎥 Provider: {video_provider}")
    print("="*60)
    
    # Load conversation data
    try:
        conversation_data = load_conversation_from_file(conversation_file)
        print(f"✅ Loaded conversation: {conversation_data['episode_info']['title']}")
        print(f"📊 Messages: {len(conversation_data['conversation'])}")
        print(f"👥 Participants: {', '.join([p['name'] for p in conversation_data['participants']])}")
    except Exception as e:
        print(f"❌ Error loading conversation file: {e}")
        return None
    
    # Reconstruct messages
    messages = reconstruct_messages_from_conversation_data(conversation_data)
    participant_details = conversation_data['participants']
    
    # Initialize video pipeline
    print(f"\n🎨 Initializing video pipeline...")
    
    # Use a capable model for storyboarding
    storyboard_model_client = OllamaChatCompletionClient(model="llama3.2")
    
    try:
        video_pipeline = VideoPipeline(
            storyboard_model_client=storyboard_model_client,
            video_provider=video_provider
        )

        # Generate video
        results = await video_pipeline.generate_full_video(messages, participant_details)

        # Update results with episode info
        results['episode_info'] = conversation_data['episode_info']
        results['source_conversation_file'] = conversation_file
        
        # Display results
        print("\n" + "="*60)
        print("🎉 VIDEO GENERATION COMPLETE!")
        print("="*60)
        print(f"Episode: {results['episode_info']['title']}")
        print(f"Duration: {results['total_duration']} seconds")
        print(f"Segments: {results['total_segments']}")
        print(f"Storyboard: {results['storyboard_file']}")
        
        if video_provider == "mock":
            print(f"📝 Mock files generated (for testing)")
        else:
            print(f"🎥 Video files: {len([f for f in results['video_files'] if f])} generated")
        
        # Show scene breakdown
        print(f"\n🎬 SCENE BREAKDOWN:")
        for i, segment in enumerate(results['storyboard'].scene_segments):
            print(f"  Scene {i+1}: {segment.speaker}")
            print(f"    Dialogue: {segment.dialogue[:60]}...")
            print(f"    Visual: {segment.scene_description[:50]}...")
            print(f"    Duration: {segment.duration_seconds}s")
            print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        return None
        
    finally:
        await storyboard_model_client.close()

async def list_available_conversations():
    """List all available conversation files."""
    conversations_dir = Path("conversations")
    
    if not conversations_dir.exists():
        print("❌ No conversations directory found. Generate some conversations first with main.py")
        return
    
    conversation_files = list(conversations_dir.glob("*.json"))
    
    if not conversation_files:
        print("❌ No conversation files found. Generate some conversations first with main.py")
        return
    
    print("📁 AVAILABLE CONVERSATIONS:")
    print("="*50)
    
    for i, file_path in enumerate(sorted(conversation_files), 1):
        try:
            data = load_conversation_from_file(str(file_path))
            title = data['episode_info'].get('title', 'Untitled')
            messages = len(data['conversation'])
            participants = ', '.join([p['name'] for p in data['participants']])
            created = data['episode_info'].get('generated_at', 'Unknown')[:19]
            
            print(f"{i:2d}. {file_path.name}")
            print(f"    Title: {title}")
            print(f"    Participants: {participants}")
            print(f"    Messages: {messages}")
            print(f"    Created: {created}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. {file_path.name} (Error loading: {e})")

async def interactive_mode():
    """Interactive mode for selecting and generating videos."""
    print("🎬 INTERACTIVE VIDEO GENERATION")
    print("="*40)
    
    # List available conversations
    await list_available_conversations()
    
    # Get user choice
    conversations_dir = Path("conversations")
    conversation_files = sorted(list(conversations_dir.glob("*.json")))
    
    if not conversation_files:
        return
    
    while True:
        try:
            choice = input(f"\nEnter conversation number (1-{len(conversation_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(conversation_files):
                selected_file = conversation_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(conversation_files)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q'")
    
    # Get video provider choice
    providers = ["mock", "runway", "pika"]
    print(f"\nAvailable video providers:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")
    
    while True:
        try:
            provider_choice = input(f"Enter provider number (1-{len(providers)}): ").strip()
            provider_num = int(provider_choice)
            
            if 1 <= provider_num <= len(providers):
                selected_provider = providers[provider_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(providers)}")
                
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Generate video
    print(f"\n🎬 Generating video from: {selected_file.name}")
    print(f"🎥 Using provider: {selected_provider}")
    
    results = await generate_video_from_file(str(selected_file), selected_provider)
    
    if results:
        print(f"\n✅ Video generation completed!")
        print(f"📁 Check the output files in the generated directories")
    else:
        print(f"\n❌ Video generation failed")

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Generate videos from saved dating show conversations")
    parser.add_argument("--conversation", "-c", help="Path to conversation JSON file")
    parser.add_argument("--provider", "-p", default="mock", 
                       choices=["mock", "runway", "pika"],
                       help="Video generation provider (default: mock)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available conversation files")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode for selecting conversations")
    
    args = parser.parse_args()
    
    if args.list:
        asyncio.run(list_available_conversations())
    elif args.interactive:
        asyncio.run(interactive_mode())
    elif args.conversation:
        if not Path(args.conversation).exists():
            print(f"❌ Conversation file not found: {args.conversation}")
            sys.exit(1)
        
        asyncio.run(generate_video_from_file(args.conversation, args.provider))
    else:
        print("Usage examples:")
        print("  python generate_video_from_conversation.py --list")
        print("  python generate_video_from_conversation.py --interactive")
        print("  python generate_video_from_conversation.py --conversation conversations/episode_123.json")
        print("  python generate_video_from_conversation.py -c conversations/episode_123.json -p runway")

if __name__ == "__main__":
    main()
