"""
Dating Show Pipeline Main Controller
This module provides CLI interface to run each stage of the 3-stage pipeline.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from stage1_agent_setup import Stage1Pipeline
from stage2_storyboard import Stage2Pipeline
from stage3_video import Stage3Pipeline


async def run_stage1(args):
    """Run Stage 1: Agent setup and conversation generation."""
    print("=== STAGE 1: CONVERSATION GENERATION ===")
    
    pipeline = Stage1Pipeline(output_dir=args.output_dir)
    
    # Handle custom configuration if provided
    custom_config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
    
    try:
        filepath = await pipeline.run_full_stage1(
            max_turns=args.max_turns,
            participants_config=custom_config
        )
        print(f"\n✅ Stage 1 completed successfully!")
        print(f"📄 Conversation file: {filepath}")
        print(f"\n➡️  To proceed to Stage 2, run:")
        print(f"   python main.py stage2 --conversation {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        sys.exit(1)


async def run_stage2(args):
    """Run Stage 2: Storyboard generation."""
    print("=== STAGE 2: STORYBOARD GENERATION ===")
    
    if not args.conversation:
        print("❌ Error: --conversation argument is required for Stage 2")
        sys.exit(1)
    
    if not Path(args.conversation).exists():
        print(f"❌ Error: Conversation file not found: {args.conversation}")
        sys.exit(1)
    
    pipeline = Stage2Pipeline(output_dir=args.output_dir)
    
    try:
        filepath = await pipeline.run_full_stage2(args.conversation)
        print(f"\n✅ Stage 2 completed successfully!")
        print(f"📄 Storyboard file: {filepath}")
        print(f"\n➡️  To proceed to Stage 3, run:")
        print(f"   python main.py stage3 --storyboard {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Stage 2 failed: {e}")
        sys.exit(1)


async def run_stage3(args):
    """Run Stage 3: Video generation."""
    print("=== STAGE 3: VIDEO GENERATION ===")
    
    if not args.storyboard:
        print("❌ Error: --storyboard argument is required for Stage 3")
        sys.exit(1)
    
    if not Path(args.storyboard).exists():
        print(f"❌ Error: Storyboard file not found: {args.storyboard}")
        sys.exit(1)
    
    pipeline = Stage3Pipeline(output_dir=args.output_dir)
    
    try:
        video_data_filepath, video_prompts_filepath = await pipeline.run_full_stage3(args.storyboard)
        print(f"\n✅ Stage 3 completed successfully!")
        print(f"📄 Video data file: {video_data_filepath}")
        print(f"📄 Video prompts file: {video_prompts_filepath}")
        print(f"\n🎬 You can now use the video prompts with AI video generation tools!")
        return video_data_filepath, video_prompts_filepath
        
    except Exception as e:
        print(f"❌ Stage 3 failed: {e}")
        sys.exit(1)


async def run_full_pipeline(args):
    """Run all three stages sequentially."""
    print("=== RUNNING FULL 3-STAGE PIPELINE ===")
    
    # Stage 1
    print("\n" + "="*50)
    stage1_args = argparse.Namespace(
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        config=getattr(args, 'config', None)
    )
    conversation_file = await run_stage1(stage1_args)
    
    # Stage 2
    print("\n" + "="*50)
    stage2_args = argparse.Namespace(
        conversation=conversation_file,
        output_dir=args.output_dir
    )
    storyboard_file = await run_stage2(stage2_args)
    
    # Stage 3
    print("\n" + "="*50)
    stage3_args = argparse.Namespace(
        storyboard=storyboard_file,
        output_dir=args.output_dir
    )
    video_files = await run_stage3(stage3_args)
    
    print(f"\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📄 Final outputs:")
    print(f"   - Conversation: {conversation_file}")
    print(f"   - Storyboard: {storyboard_file}")
    print(f"   - Video data: {video_files[0]}")
    print(f"   - Video prompts: {video_files[1]}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dating Show 3-Stage Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run individual stages
  python main.py stage1 --max-turns 10
  python main.py stage2 --conversation pipeline_output/conversation_20241201_143022.json
  python main.py stage3 --storyboard pipeline_output/storyboard_20241201_143045.json
  
  # Run full pipeline
  python main.py full --max-turns 8
  
  # Use custom configuration
  python main.py stage1 --config custom_config.json
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='stage', help='Pipeline stage to run')
    
    # Stage 1 subcommand
    stage1_parser = subparsers.add_parser('stage1', help='Run Stage 1: Conversation generation')
    stage1_parser.add_argument('--max-turns', type=int, default=5, help='Maximum conversation turns')
    stage1_parser.add_argument('--config', default='participants_config.json', help='Path to custom configuration JSON file')
    stage1_parser.add_argument('--output-dir', default='pipeline_output', help='Output directory')
    
    # Stage 2 subcommand
    stage2_parser = subparsers.add_parser('stage2', help='Run Stage 2: Storyboard generation')
    stage2_parser.add_argument('--conversation', required=True, help='Path to conversation JSON file')
    stage2_parser.add_argument('--output-dir', default='pipeline_output', help='Output directory')
    
    # Stage 3 subcommand
    stage3_parser = subparsers.add_parser('stage3', help='Run Stage 3: Video generation')
    stage3_parser.add_argument('--storyboard', required=True, help='Path to storyboard JSON file')
    stage3_parser.add_argument('--output-dir', default='pipeline_output', help='Output directory')
    
    # Full pipeline subcommand
    full_parser = subparsers.add_parser('full', help='Run full 3-stage pipeline')
    full_parser.add_argument('--max-turns', type=int, default=5, help='Maximum conversation turns')
    full_parser.add_argument('--config', help='Path to custom configuration JSON file')
    full_parser.add_argument('--output-dir', default='pipeline_output', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.stage:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate stage
    if args.stage == 'stage1':
        asyncio.run(run_stage1(args))
    elif args.stage == 'stage2':
        asyncio.run(run_stage2(args))
    elif args.stage == 'stage3':
        asyncio.run(run_stage3(args))
    elif args.stage == 'full':
        asyncio.run(run_full_pipeline(args))
    else:
        print(f"❌ Unknown stage: {args.stage}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
