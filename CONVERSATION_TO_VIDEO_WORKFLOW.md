# Conversation-to-Video Workflow

This enhanced dating show system now supports a **decoupled workflow** where conversations are saved to files and can be processed into videos separately.

## 🎯 **Why This Approach?**

- **Reproducible**: Generate videos from the same conversation multiple times
- **Reviewable**: Examine and edit conversations before expensive video generation
- **Flexible**: Try different video providers/settings on the same conversation
- **Recoverable**: If video generation fails, conversation is preserved
- **Batchable**: Process multiple conversations efficiently

## 📁 **Workflow Overview**

```
Step 1: Generate Conversation → Step 2: Save to File → Step 3: Generate Video
    (main.py)                    (conversations/*.json)    (video pipeline)
```

## 🚀 **Quick Start**

### 1. Generate a Conversation

```bash
python main.py
```

This creates a conversation file in `conversations/dating_show_episode_TIMESTAMP.json`

### 2. List Available Conversations

```bash
python conversation_manager.py list
python conversation_manager.py list --detailed  # More info
```

### 3. Generate Video from Conversation

```bash
# Interactive mode (recommended for beginners)
python generate_video_from_conversation.py --interactive

# Or specify file directly
python generate_video_from_conversation.py --conversation conversations/dating_show_episode_20241226_143022.json

# With specific video provider
python generate_video_from_conversation.py -c conversations/episode.json -p runway
```

### 4. Test the Full Workflow

```bash
python test_full_workflow.py
```

## 📋 **Available Commands**

### Conversation Management

```bash
# List all conversations
python conversation_manager.py list

# Show detailed conversation info
python conversation_manager.py show conversations/episode.json

# Export conversation as text
python conversation_manager.py export conversations/episode.json

# Delete a conversation
python conversation_manager.py delete conversations/episode.json
```

### Video Generation

```bash
# List available conversations
python generate_video_from_conversation.py --list

# Interactive selection
python generate_video_from_conversation.py --interactive

# Generate with specific provider
python generate_video_from_conversation.py -c file.json -p mock     # Testing
python generate_video_from_conversation.py -c file.json -p runway   # High quality
python generate_video_from_conversation.py -c file.json -p pika     # Balanced
```

## 📊 **File Structure**

After running the workflow, you'll have:

```
dating-show/
├── conversations/           # Saved conversations
│   ├── dating_show_episode_20241226_143022.json
│   └── sample_episode_first_sparks.json
├── pipeline_output/         # Storyboards
│   └── First_Sparks_storyboard.json
├── video_prompts/          # Generated prompts for video APIs
│   └── First_Sparks_20241226_143022_prompts.json
└── generated_videos/       # Video outputs
    ├── mock_segment_0.txt  # Mock files for testing
    └── runway_segment_1.mp4 # Real videos when using APIs
```

## 🎨 **Conversation File Format**

Each conversation file contains:

```json
{
  "episode_info": {
    "title": "The First Meeting",
    "season": 1,
    "episode": 1,
    "description": "Alex and Sarah meet for the first time",
    "generated_at": "2024-12-26T14:30:22",
    "total_messages": 10
  },
  "participants": [
    {
      "name": "Alex",
      "age": 28,
      "personality_traits": ["confident", "adventurous"],
      "agent_type": "participant"
    }
  ],
  "director": {
    "name": "Director",
    "agent_type": "director"
  },
  "conversation": [
    {
      "speaker": "Director",
      "content": "Welcome to the show!",
      "clean_content": "Welcome to the show!",
      "message_type": "AssistantMessage"
    }
  ],
  "metadata": {
    "participant_speak_counts": { "Alex": 4, "Sarah": 3 },
    "director_interventions": 2,
    "total_duration_estimate": 150
  }
}
```

## 🎬 **Video Generation Pipeline**

The three-stage pipeline:

1. **Conversation** (Already generated and saved)
2. **Storyboarding** (LLM analyzes conversation and creates cinematic plan)
3. **Video Generation** (Multimodal LLM generates actual video segments)

### Storyboard Output

```json
{
  "episode_title": "First Sparks",
  "total_duration": 300,
  "overall_mood": "romantic_hopeful",
  "scene_segments": [
    {
      "speaker": "Director",
      "dialogue": "Welcome everyone!",
      "duration_seconds": 15.0,
      "scene_description": "Wide shot of luxurious villa entrance",
      "camera_angle": "wide_establishing_shot",
      "mood": "welcoming_dramatic",
      "lighting": "golden_hour_dramatic"
    }
  ]
}
```

## ⚙️ **Video Providers**

- **mock**: For testing (creates text files instead of videos)
- **runway**: RunwayML API (high quality, expensive)
- **pika**: Pika Labs API (balanced quality/cost)

To use real video generation APIs, you'll need to:

1. Sign up for the service
2. Get API keys
3. Update the `MultimodalVideoGenerator` class with actual API calls

## 🔧 **Configuration**

### Model Settings

Edit the model clients in the scripts:

- **Conversation**: Uses Ollama (llama3.2) + OpenAI (gemini-2.0-flash)
- **Storyboarding**: Uses OpenAI (gpt-4) for best quality
- **Video Generation**: Uses external APIs (Runway/Pika)

### Episode Settings

Customize in `main.py`:

```python
episode_info = {
    "title": "Your Episode Title",
    "season": 1,
    "episode": 1,
    "description": "Episode description"
}
```

## 🐛 **Troubleshooting**

### Common Issues

1. **No conversations found**: Run `python main.py` first
2. **Video generation fails**: Check API keys and provider settings
3. **File not found**: Use full paths or check working directory
4. **JSON parse errors**: Check conversation file format

### Debug Steps

1. Test conversation generation: `python main.py`
2. Verify file was created: `python conversation_manager.py list`
3. Test with mock provider: `python generate_video_from_conversation.py -p mock`
4. Check storyboard output in `pipeline_output/`

## 🎯 **Next Steps**

1. **Generate your first conversation**: `python main.py`
2. **Test the pipeline**: `python test_full_workflow.py`
3. **Explore saved conversations**: `python conversation_manager.py list --detailed`
4. **Try video generation**: `python generate_video_from_conversation.py --interactive`

The system is now fully decoupled and ready for production use! 🎉
