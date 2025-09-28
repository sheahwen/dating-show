"""
Utilities for saving, loading, and managing conversation files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from utils import extract_content_without_thoughts

def save_conversation_to_file(messages, participants, director, episode_info: Optional[Dict] = None) -> str:
    """
    Save conversation and metadata to JSON file.
    
    Args:
        messages: List of conversation messages
        participants: List of participant agents
        director: Director agent
        episode_info: Optional episode metadata
        
    Returns:
        str: Path to the saved conversation file
    """
    # Create conversations directory
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dating_show_episode_{timestamp}.json"
    filepath = conversations_dir / filename
    
    # Prepare conversation data
    conversation_data = {
        "episode_info": episode_info or {
            "title": f"Dating Show Episode {timestamp}",
            "generated_at": datetime.now().isoformat(),
            "total_messages": len(messages)
        },
        "participants": [p.get_details() for p in participants],
        "director": director.get_details(),
        "conversation": [
            {
                "speaker": msg.source,
                "content": msg.content,
                "clean_content": extract_content_without_thoughts(msg.content),
                "message_type": type(msg).__name__
            }
            for msg in messages
        ],
        "metadata": {
            "participant_speak_counts": {
                p.name: sum(1 for msg in messages if msg.source == p.name) 
                for p in participants
            },
            "director_interventions": sum(1 for msg in messages if msg.source == "Director"),
            "total_duration_estimate": len(messages) * 15  # Rough estimate: 15 seconds per message
        }
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Conversation saved to: {filepath}")
    return str(filepath)


def load_conversation_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load conversation from JSON file.
    
    Args:
        filepath: Path to the conversation JSON file
        
    Returns:
        Dict containing the conversation data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_conversations_directory() -> Path:
    """Get the conversations directory path, creating it if it doesn't exist."""
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)
    return conversations_dir


def list_conversation_files() -> List[Path]:
    """Get a list of all conversation files, sorted by modification time."""
    conversations_dir = get_conversations_directory()
    conversation_files = list(conversations_dir.glob("*.json"))
    return sorted(conversation_files, key=lambda f: f.stat().st_mtime, reverse=True)


def get_conversation_summary(filepath: str) -> Dict[str, Any]:
    """
    Get a summary of a conversation file without loading the full content.
    
    Args:
        filepath: Path to the conversation file
        
    Returns:
        Dict with summary information
    """
    try:
        data = load_conversation_from_file(filepath)
        
        return {
            "filepath": filepath,
            "filename": Path(filepath).name,
            "title": data['episode_info'].get('title', 'Untitled'),
            "participants": [p['name'] for p in data['participants']],
            "message_count": len(data['conversation']),
            "created_at": data['episode_info'].get('generated_at', 'Unknown'),
            "file_size_kb": Path(filepath).stat().st_size / 1024,
            "metadata": data.get('metadata', {})
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "filename": Path(filepath).name,
            "error": str(e)
        }


def validate_conversation_file(filepath: str) -> bool:
    """
    Validate that a conversation file has the expected structure.
    
    Args:
        filepath: Path to the conversation file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        data = load_conversation_from_file(filepath)
        
        # Check required top-level keys
        required_keys = ['episode_info', 'participants', 'director', 'conversation']
        if not all(key in data for key in required_keys):
            return False
        
        # Check episode_info structure
        episode_info = data['episode_info']
        if not isinstance(episode_info, dict) or 'title' not in episode_info:
            return False
        
        # Check participants structure
        participants = data['participants']
        if not isinstance(participants, list) or len(participants) == 0:
            return False
        
        # Check conversation structure
        conversation = data['conversation']
        if not isinstance(conversation, list):
            return False
        
        # Check each message has required fields
        for msg in conversation:
            if not all(key in msg for key in ['speaker', 'content', 'clean_content']):
                return False
        
        return True
        
    except Exception:
        return False


def backup_conversation_file(filepath: str) -> str:
    """
    Create a backup copy of a conversation file.
    
    Args:
        filepath: Path to the original conversation file
        
    Returns:
        str: Path to the backup file
    """
    original_path = Path(filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{original_path.stem}_backup_{timestamp}.json"
    backup_path = original_path.parent / backup_filename
    
    # Copy the file
    import shutil
    shutil.copy2(filepath, backup_path)
    
    print(f"📋 Backup created: {backup_path}")
    return str(backup_path)


def cleanup_old_conversations(keep_count: int = 10) -> List[str]:
    """
    Clean up old conversation files, keeping only the most recent ones.
    
    Args:
        keep_count: Number of most recent conversations to keep
        
    Returns:
        List of deleted file paths
    """
    conversation_files = list_conversation_files()
    
    if len(conversation_files) <= keep_count:
        print(f"📁 Only {len(conversation_files)} conversations found, nothing to clean up")
        return []
    
    # Files to delete (oldest ones)
    files_to_delete = conversation_files[keep_count:]
    deleted_files = []
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_files.append(str(file_path))
            print(f"🗑️  Deleted: {file_path.name}")
        except Exception as e:
            print(f"❌ Error deleting {file_path.name}: {e}")
    
    print(f"🧹 Cleanup complete: {len(deleted_files)} files deleted, {keep_count} kept")
    return deleted_files
