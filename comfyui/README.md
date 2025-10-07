# ComfyUI Workflows for ExcuseMyFrench

This directory contains ComfyUI workflow files for generating character images in the ExcuseMyFrench video pipeline.

## Workflow Files

### character_generation.json
General-purpose workflow for generating character images with Wan 2.2 models.

**Parameters:**
- Model: wan_2.2_base.safetensors
- Resolution: 1080x1920 (vertical video format)
- Steps: 30
- CFG Scale: 7.5
- Sampler: euler_ancestral

**Use Cases:**
- Butcher (French bulldog) character images
- Default workflow for most character generation

### nutsy_generation.json
Optimized workflow for generating the Nutsy squirrel character.

**Parameters:**
- Model: wan_2.2_base.safetensors
- Resolution: 1080x1920
- Steps: 30
- CFG Scale: 8.0 (higher for more prompt adherence)
- Sampler: euler_ancestral

**Use Cases:**
- Nutsy (squirrel) character images
- Cartoon/anthropomorphic style characters

## Usage

### Loading in ComfyUI Interface

1. Start ComfyUI server
2. Open http://127.0.0.1:8188 in browser
3. Click "Load" button
4. Navigate to this directory
5. Select workflow JSON file

### Using with Python API

```python
from scripts.comfyui_integration import ComfyUIClient

client = ComfyUIClient()

# Generate Butcher image
params = {
    'prompt': 'French bulldog, happy expression, portrait, detailed',
    'negative_prompt': 'blurry, low quality',
    'seed': 42
}

images = client.generate_image(
    workflow_path='comfyui/workflows/character_generation.json',
    params=params,
    output_path='output/butcher_happy.png'
)

# Generate Nutsy image
params = {
    'prompt': 'cute squirrel character, excited, cartoon style, detailed',
    'negative_prompt': 'blurry, dark, creepy',
    'seed': 123
}

images = client.generate_image(
    workflow_path='comfyui/workflows/nutsy_generation.json',
    params=params,
    output_path='output/nutsy_excited.png'
)
```

### Command Line Usage

```bash
# Generate with character workflow
python scripts/comfyui_integration.py \
    --workflow comfyui/workflows/character_generation.json \
    --prompt "French bulldog, happy expression" \
    --output butcher_happy.png

# Generate with Nutsy workflow
python scripts/comfyui_integration.py \
    --workflow comfyui/workflows/nutsy_generation.json \
    --prompt "cute squirrel character, excited" \
    --output nutsy_excited.png
```

## Customizing Workflows

### Creating New Workflows

1. Open ComfyUI web interface
2. Design your workflow with desired nodes
3. Test the workflow
4. Click "Save" to export as JSON
5. Save to this directory with descriptive name

### Modifying Existing Workflows

You can edit the JSON files directly or:

1. Load workflow in ComfyUI interface
2. Make changes
3. Save with new name or overwrite

### Important Nodes

- **CheckpointLoaderSimple**: Loads the AI model
- **CLIPTextEncode**: Encodes text prompts
- **EmptyLatentImage**: Sets image dimensions
- **KSampler**: Main generation node (steps, seed, cfg)
- **VAEDecode**: Converts latent to image
- **SaveImage**: Saves output

## Emotion-Specific Prompts

### Butcher (French Bulldog)

```python
emotions = {
    'happy': 'smiling, joyful expression, cheerful',
    'sad': 'sad expression, downcast, melancholy',
    'sarcastic': 'smirking, knowing look, raised eyebrow',
    'angry': 'angry expression, fierce, furrowed brow',
    'confused': 'confused expression, puzzled, tilted head',
    'surprised': 'surprised expression, wide eyes, amazed',
    'neutral': 'neutral expression, calm, relaxed'
}
```

### Nutsy (Squirrel)

```python
emotions = {
    'happy': 'smiling, joyful, cheerful demeanor',
    'excited': 'excited, energetic, enthusiastic, wide eyes',
    'sad': 'sad, downcast, droopy tail',
    'nervous': 'nervous, anxious, worried expression',
    'confused': 'confused, puzzled, questioning look',
    'surprised': 'surprised, shocked, wide eyes',
    'neutral': 'neutral expression, calm, friendly'
}
```

## Tips for Best Results

### Quality Settings

- **Steps**: 25-35 for good quality (higher = slower but better)
- **CFG Scale**: 7.0-9.0 for balance (higher = closer to prompt)
- **Resolution**: Use 1080x1920 for vertical video format
- **Seed**: Use specific seeds for reproducibility

### Prompt Engineering

**Good Prompts:**
- Descriptive: "French bulldog with sarcastic expression, portrait"
- Specific details: "detailed fur texture, expressive eyes"
- Style keywords: "professional photography, high quality"

**Avoid:**
- Too vague: "dog"
- Conflicting terms: "happy and sad"
- Too many details: (keeps prompts focused)

### Negative Prompts

Always include to avoid common issues:
- "blurry, low quality, distorted"
- "extra limbs, bad anatomy"
- "watermark, text, signature"
- "multiple animals, duplicate"

## Performance Tips

### For Faster Generation

1. Reduce steps to 20-25
2. Use smaller resolution (but maintain aspect ratio)
3. Use faster samplers (euler, heun)
4. Keep server running between generations

### For Better Quality

1. Increase steps to 35-40
2. Use full resolution (1080x1920)
3. Try different samplers (dpm++ 2m, ddim)
4. Fine-tune CFG scale
5. Experiment with seeds

## Troubleshooting

### Workflow Fails to Load

- Check JSON syntax is valid
- Verify all node types are available
- Ensure custom nodes are installed

### Generated Images Look Wrong

- Adjust prompt for clarity
- Modify CFG scale (7-9 range)
- Try different seeds
- Check negative prompt

### Out of Memory

- Reduce image resolution
- Lower batch size to 1
- Enable --lowvram mode in ComfyUI
- Close other applications

## Next Steps

1. Test workflows with sample prompts
2. Generate images for each emotion
3. Add generated images to library
4. Integrate with main video pipeline

See [docs/comfyui_setup.md](../docs/comfyui_setup.md) for detailed setup instructions.
