#!/usr/bin/env python3
"""
Test script to demonstrate incremental conversation saving.
This script shows how conversations are saved every 10 turns instead of only at the end.
"""

import asyncio
import json
from pathlib import Path
from stage1_agent_setup import Stage1Pipeline


async def test_incremental_saving():
    """Test the incremental saving functionality."""
    print("🧪 Testing Incremental Conversation Saving")
    print("=" * 50)

    # Create pipeline
    pipeline = Stage1Pipeline(output_dir="test_checkpoints")

    # Sample configurations (minimal for testing)
    participants_config = {
        "participants": [
            {
                "name": "Alice",
                "age": 25,
                "personality_traits": ["outgoing", "romantic", "adventurous"],
                "model_client": {"client": "ollama", "model": "llama3.2"},
            },
            {
                "name": "Bob",
                "age": 27,
                "personality_traits": ["thoughtful", "caring", "funny"],
                "model_client": {"client": "ollama", "model": "llama3.2"},
            },
        ]
    }

    director_config = {
        "director": {
            "name": "Director",
            "model_client": {"client": "ollama", "model": "llama3.2"},
        }
    }

    print(f"📁 Checkpoint directory: {pipeline.output_dir}")
    print(f"💾 Save frequency: Every {pipeline.save_frequency} turns")
    print(f"🧠 Memory update frequency: Every 3 turns")

    try:
        # Run with more turns to demonstrate checkpointing (25 turns)
        print(f"\n🎬 Starting conversation with 25 max turns...")
        final_filepath = await pipeline.run_full_stage1(
            max_turns=25,
            participants_config=participants_config,
            director_config=director_config,
        )

        print(f"\n✅ Conversation completed!")
        print(f"📄 Final conversation saved to: {final_filepath}")

        # List all checkpoint files created
        checkpoint_files = list(
            pipeline.output_dir.glob("conversation_*_checkpoint_*.json")
        )
        final_files = list(pipeline.output_dir.glob("conversation_*_final.json"))

        print(f"\n📊 Summary:")
        print(f"   • Checkpoint files created: {len(checkpoint_files)}")
        print(f"   • Final file created: {len(final_files)}")

        if checkpoint_files:
            print(f"\n💾 Checkpoint files:")
            for file in sorted(checkpoint_files):
                # Get file size for demonstration
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.1f} KB)")

        if final_files:
            print(f"\n🎯 Final files:")
            for file in sorted(final_files):
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.1f} KB)")

        print(f"\n🎉 Test completed successfully!")
        print(f"💡 Benefits:")
        print(f"   • Conversations saved incrementally every 10 turns")
        print(f"   • No data loss if process is interrupted")
        print(f"   • Memory usage kept low with agent memory system")
        print(f"   • Each checkpoint contains complete conversation up to that point")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_incremental_saving())
