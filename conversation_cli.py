#!/usr/bin/env python3
"""
Utility script for managing saved conversations.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from conversation_utils import load_conversation_from_file

def list_conversations(detailed: bool = False):
    """List all saved conversations."""
    conversations_dir = Path("conversations")
    
    if not conversations_dir.exists():
        print("❌ No conversations directory found.")
        return
    
    conversation_files = sorted(list(conversations_dir.glob("*.json")))
    
    if not conversation_files:
        print("❌ No conversation files found.")
        return
    
    print("📁 SAVED CONVERSATIONS:")
    print("="*60)
    
    for i, file_path in enumerate(conversation_files, 1):
        try:
            data = load_conversation_from_file(str(file_path))
            
            # Basic info
            title = data['episode_info'].get('title', 'Untitled')
            messages = len(data['conversation'])
            participants = [p['name'] for p in data['participants']]
            created = data['episode_info'].get('generated_at', 'Unknown')[:19]
            file_size = file_path.stat().st_size / 1024  # KB
            
            print(f"{i:2d}. {file_path.name}")
            print(f"    📺 Title: {title}")
            print(f"    👥 Participants: {', '.join(participants)}")
            print(f"    💬 Messages: {messages}")
            print(f"    📅 Created: {created}")
            print(f"    📁 Size: {file_size:.1f} KB")
            
            if detailed:
                # Show metadata
                metadata = data.get('metadata', {})
                speak_counts = metadata.get('participant_speak_counts', {})
                director_interventions = metadata.get('director_interventions', 0)
                duration_est = metadata.get('total_duration_estimate', 0)
                
                print(f"    📊 Speaking distribution:")
                for participant, count in speak_counts.items():
                    percentage = (count / messages * 100) if messages > 0 else 0
                    print(f"       {participant}: {count} messages ({percentage:.1f}%)")
                
                print(f"    🎬 Director interventions: {director_interventions}")
                print(f"    ⏱️  Estimated duration: {duration_est // 60}m {duration_est % 60}s")
                
                # Show conversation preview
                print(f"    🗨️  Preview:")
                for msg in data['conversation'][:3]:  # First 3 messages
                    speaker = msg['speaker']
                    content = msg['clean_content'][:50] + "..." if len(msg['clean_content']) > 50 else msg['clean_content']
                    print(f"       {speaker}: {content}")
                if len(data['conversation']) > 3:
                    print(f"       ... and {len(data['conversation']) - 3} more messages")
            
            print()
            
        except Exception as e:
            print(f"{i:2d}. {file_path.name} (❌ Error: {e})")
            print()

def show_conversation_details(conversation_file: str):
    """Show detailed information about a specific conversation."""
    try:
        data = load_conversation_from_file(conversation_file)
        
        print(f"📺 CONVERSATION DETAILS")
        print("="*50)
        
        # Episode info
        episode_info = data['episode_info']
        print(f"Title: {episode_info.get('title', 'Untitled')}")
        print(f"Season: {episode_info.get('season', 'N/A')}")
        print(f"Episode: {episode_info.get('episode', 'N/A')}")
        print(f"Description: {episode_info.get('description', 'No description')}")
        print(f"Generated: {episode_info.get('generated_at', 'Unknown')}")
        
        # Participants
        print(f"\n👥 PARTICIPANTS:")
        for participant in data['participants']:
            traits = ', '.join(participant.get('personality_traits', []))
            print(f"  • {participant['name']} (age {participant['age']})")
            print(f"    Traits: {traits}")
        
        # Director
        director = data['director']
        print(f"\n🎬 DIRECTOR:")
        print(f"  • {director['name']}")
        
        # Conversation stats
        metadata = data.get('metadata', {})
        speak_counts = metadata.get('participant_speak_counts', {})
        total_messages = len(data['conversation'])
        
        print(f"\n📊 CONVERSATION STATISTICS:")
        print(f"  Total messages: {total_messages}")
        print(f"  Director interventions: {metadata.get('director_interventions', 0)}")
        print(f"  Estimated duration: {metadata.get('total_duration_estimate', 0)} seconds")
        
        print(f"\n💬 SPEAKING DISTRIBUTION:")
        for speaker, count in speak_counts.items():
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            print(f"  {speaker}: {count} messages ({percentage:.1f}%)")
        
        # Full conversation
        print(f"\n🗨️  FULL CONVERSATION:")
        print("-" * 50)
        for i, msg in enumerate(data['conversation'], 1):
            speaker = msg['speaker']
            content = msg['clean_content']
            print(f"{i:2d}. {speaker}: {content}")
            print()
        
    except Exception as e:
        print(f"❌ Error loading conversation: {e}")

def export_conversation_text(conversation_file: str, output_file: str = None):
    """Export conversation as plain text."""
    try:
        data = load_conversation_from_file(conversation_file)
        
        if not output_file:
            # Generate output filename
            input_path = Path(conversation_file)
            output_file = input_path.parent / f"{input_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            episode_info = data['episode_info']
            f.write(f"Dating Show Episode: {episode_info.get('title', 'Untitled')}\n")
            f.write(f"Generated: {episode_info.get('generated_at', 'Unknown')}\n")
            f.write("="*60 + "\n\n")
            
            # Write participants
            f.write("PARTICIPANTS:\n")
            for participant in data['participants']:
                traits = ', '.join(participant.get('personality_traits', []))
                f.write(f"• {participant['name']} (age {participant['age']}): {traits}\n")
            f.write("\n")
            
            # Write conversation
            f.write("CONVERSATION:\n")
            f.write("-" * 40 + "\n\n")
            
            for msg in data['conversation']:
                speaker = msg['speaker']
                content = msg['clean_content']
                f.write(f"{speaker}: {content}\n\n")
        
        print(f"✅ Conversation exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error exporting conversation: {e}")

def delete_conversation(conversation_file: str, confirm: bool = False):
    """Delete a conversation file."""
    file_path = Path(conversation_file)
    
    if not file_path.exists():
        print(f"❌ File not found: {conversation_file}")
        return
    
    if not confirm:
        # Show conversation details first
        try:
            data = load_conversation_from_file(conversation_file)
            title = data['episode_info'].get('title', 'Untitled')
            print(f"📺 About to delete: {title}")
            print(f"📁 File: {conversation_file}")
            
            response = input("Are you sure you want to delete this conversation? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    
    try:
        file_path.unlink()
        print(f"✅ Deleted: {conversation_file}")
    except Exception as e:
        print(f"❌ Error deleting file: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage saved dating show conversations")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all conversations')
    list_parser.add_argument('--detailed', '-d', action='store_true',
                           help='Show detailed information')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show conversation details')
    show_parser.add_argument('file', help='Conversation file path')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export conversation as text')
    export_parser.add_argument('file', help='Conversation file path')
    export_parser.add_argument('--output', '-o', help='Output file path')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a conversation')
    delete_parser.add_argument('file', help='Conversation file path')
    delete_parser.add_argument('--force', '-f', action='store_true',
                             help='Skip confirmation')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_conversations(detailed=args.detailed)
    elif args.command == 'show':
        show_conversation_details(args.file)
    elif args.command == 'export':
        export_conversation_text(args.file, args.output)
    elif args.command == 'delete':
        delete_conversation(args.file, confirm=args.force)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
