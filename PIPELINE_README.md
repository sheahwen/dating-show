# Dating Show 3-Stage Pipeline

This project implements a 3-stage pipeline for generating dating show content:

1. **Stage 1**: Generate participants, directors, and conversations
2. **Stage 2**: Generate storyboards from conversations using LLM
3. **Stage 3**: Generate video prompts from storyboards using LLM

## Architecture

The pipeline is designed to be modular and controllable, allowing you to review and approve outputs at each stage before proceeding to the next.

### Files Structure

```
dating-show/
├── main.py                 # CLI controller for all pipeline stages
├── models.py               # Pydantic models for data schemas
├── stage1_agent_setup.py   # Stage 1: Agent setup and conversation generation
├── stage2_storyboard.py    # Stage 2: Storyboard generation from conversations
├── stage3_video.py         # Stage 3: Video prompt generation from storyboards
├── agent.py                # Agent classes (existing)
├── dating_show_chat.py     # Chat orchestration (existing)
├── utils.py                # Utility functions (existing)
├── participants_config.json     # Configuration for participants
├── pipeline_output/        # Output directory for JSON files
└── generated_videos/       # Output directory for video prompts
```

## Installation

1. Install dependencies:

```bash
uv sync
```

2. Set up your environment variables in `.env`:

```bash
# Add your API keys as needed
OPENAI_API_KEY=your_openai_key
# Other model provider keys...
```

## Usage

### CLI Commands

The main interface is through `main.py` with subcommands for each stage:

```bash
# Show help
python main.py --help

# Run individual stages
python main.py stage1 --max-turns 10
python main.py stage2 --conversation pipeline_output/conversation_20241201_143022.json
python main.py stage3 --storyboard pipeline_output/storyboard_20241201_143045.json

# Run full pipeline (all stages sequentially)
python main.py full --max-turns 8

# Use custom configuration
python main.py stage1 --config participants_config.json
```

### Stage 1: Conversation Generation

Generates participants and directors, then runs a conversation between them.

**Input**: Configuration (optional)
**Output**: `conversation_TIMESTAMP.json`

```bash
# Default participants (Alex & Sarah)
python main.py stage1

# Custom configuration
python main.py stage1 --config participants_config.json --max-turns 10

# Run independently
python stage1_agent_setup.py
```

**Output Structure**:

- Participant information (names, ages, personality traits)
- Director information
- Complete conversation messages
- Metadata (timestamps, turn counts, etc.)

### Stage 2: Storyboard Generation

Takes a conversation file and generates a detailed storyboard using LLM.

**Input**: Conversation JSON file
**Output**: `storyboard_TIMESTAMP.json`

```bash
python main.py stage2 --conversation pipeline_output/conversation_20241201_143022.json

# Run independently
python stage2_storyboard.py --conversation pipeline_output/conversation_20241201_143022.json
```

**Output Structure**:

- Episode title and theme
- 8-12 scenes with detailed descriptions
- Duration, characters, settings, moods
- Key dialogue and visual notes

### Stage 3: Video Generation

Takes a storyboard file and generates detailed video prompts for AI video generation tools.

**Input**: Storyboard JSON file
**Output**:

- `video_data_TIMESTAMP.json` (structured data)
- `video_prompts_TIMESTAMP.txt` (human-readable prompts)

```bash
python main.py stage3 --storyboard pipeline_output/storyboard_20241201_143045.json

# Run independently
python stage3_video.py --storyboard pipeline_output/storyboard_20241201_143045.json
```

**Output Structure**:

- Video segments with detailed prompts
- Technical specifications (resolution, FPS)
- Style notes and visual descriptions
- Ready-to-use prompts for AI video tools

## Data Models

The pipeline uses Pydantic models for type safety and validation:

- `ConversationData`: Complete conversation with participants and messages
- `StoryboardData`: Structured storyboard with scenes and metadata
- `VideoData`: Video generation data with prompts and specifications

## Customization

### Custom Participants

Create a JSON configuration file:

```json
{
  "participants": [
    {
      "name": "Custom Name",
      "age": 30,
      "personality_traits": ["trait1", "trait2", "trait3"]
    }
  ],
  "director": {
    "name": "Custom Director"
  }
}
```

### Model Configuration

Edit the model clients in `stage1_agent_setup.py`:

```python
# Change these lines to use different models
p1_model_client = OllamaChatCompletionClient(model="llama3.2:1b")
p2_model_client = OllamaChatCompletionClient(model="llama3.2")
director_model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash")
```

## Output Files

### Conversation File Example

```json
{
  "participants": [...],
  "director": {...},
  "messages": [...],
  "created_at": "2024-12-01T14:30:22",
  "total_turns": 8
}
```

### Storyboard File Example

```json
{
  "title": "First Connections",
  "total_duration_seconds": 300,
  "scenes": [
    {
      "scene_number": 1,
      "title": "Villa Introduction",
      "description": "Wide establishing shot...",
      "duration_seconds": 25.0,
      "characters_involved": ["Alex", "Sarah"],
      "setting": "Villa entrance",
      "mood": "Anticipatory"
    }
  ]
}
```

### Video Prompts File Example

```
SEGMENT 1 (Scene 1)
Duration: 25s
Characters: Alex, Sarah
Setting: Villa entrance
Mood: Anticipatory

VIDEO PROMPT:
Wide establishing shot of a luxurious Mediterranean villa at golden hour. Camera slowly zooms in as two attractive young people, Alex (confident, 28) and Sarah (artistic, 25), approach the entrance separately. Cinematic lighting with warm sunset tones. Reality TV style with multiple camera angles. 4K resolution, smooth camera movement.
```

## AI Video Generation

The generated video prompts are optimized for AI video generation tools:

- **Runway ML**: Use the detailed prompts directly
- **Pika Labs**: Copy prompts to their interface
- **OpenAI Sora**: Adapt prompts for their format
- **Stable Video Diffusion**: Use with appropriate parameters

## Troubleshooting

1. **Missing dependencies**: Run `uv sync` to install all required packages
2. **API key errors**: Check your `.env` file has the correct API keys
3. **File not found**: Ensure you're using the correct file paths from previous stages
4. **JSON parsing errors**: Check that LLM responses are valid JSON (retry if needed)

## Development

To extend the pipeline:

1. **Add new stages**: Create new `stageN_*.py` files following the existing pattern
2. **Modify models**: Update `models.py` with new Pydantic schemas
3. **Change LLM prompts**: Edit the prompt generation methods in each stage
4. **Add new output formats**: Extend the save methods in each pipeline class

## Examples

### Quick Start

```bash
# Generate everything in one go
python main.py full --max-turns 5

# Review outputs in pipeline_output/ and generated_videos/
```

### Step-by-step with Review

```bash
# Step 1: Generate conversation
python main.py stage1 --max-turns 8
# Review: pipeline_output/conversation_TIMESTAMP.json

# Step 2: Generate storyboard (replace TIMESTAMP)
python main.py stage2 --conversation pipeline_output/conversation_TIMESTAMP.json
# Review: pipeline_output/storyboard_TIMESTAMP.json

# Step 3: Generate video prompts (replace TIMESTAMP)
python main.py stage3 --storyboard pipeline_output/storyboard_TIMESTAMP.json
# Review: generated_videos/video_prompts_TIMESTAMP.txt
```

This pipeline gives you full control over each stage while maintaining the ability to run everything automatically when needed.
