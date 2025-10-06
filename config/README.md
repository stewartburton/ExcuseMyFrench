# Configuration Setup

## Quick Start

1. **Open the `.env` file** in this directory
2. **Paste your API keys** into the appropriate fields:
   ```bash
   ELEVENLABS_API_KEY=your-actual-key-here
   ELEVENLABS_VOICE_BUTCHER=your-voice-id-here
   ELEVENLABS_VOICE_NUTSY=your-voice-id-here
   HF_TOKEN=hf_your-token-here
   ANTHROPIC_API_KEY=sk-ant-your-key-here  # or use OPENAI_API_KEY
   ```
3. **Save the file**
4. **Verify** that `.env` is listed in `.gitignore` (it is!)

## Required Keys to Start Development

**Minimum to begin:**
- ✅ `ELEVENLABS_API_KEY` - For voice synthesis
- ✅ `ELEVENLABS_VOICE_BUTCHER` - Voice ID for Butcher character
- ✅ `ELEVENLABS_VOICE_NUTSY` - Voice ID for Nutsy character
- ✅ `HF_TOKEN` - Hugging Face token for model downloads
- ✅ `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` - For script generation

**Can be added later:**
- Instagram/Meta API keys (for posting)
- Music generation API keys (Mubert/Soundverse)
- TikTok API keys (optional)

## Files in This Directory

- **`.env.example`** - Template showing all available configuration options (committed to git)
- **`.env`** - Your actual API keys and secrets (**NOT committed to git**, safe to edit)
- **`README.md`** - This file

## Security Notes

- ⚠️ **NEVER commit `.env` to version control**
- ⚠️ The `.env` file is already in `.gitignore` - don't remove it
- ⚠️ Don't share your `.env` file or paste keys in public issues
- ✅ Use `.env.example` as a template when sharing configuration needs

## Verification

To verify your configuration is loaded correctly, you can run:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv('config/.env'); print('✓ Config loaded' if os.getenv('ELEVENLABS_API_KEY') else '✗ Keys not found')"
```

## Need Help?

- ElevenLabs setup: https://elevenlabs.io/app/settings/api-keys
- Hugging Face tokens: https://huggingface.co/settings/tokens
- OpenAI API: https://platform.openai.com/api-keys
- Anthropic API: https://console.anthropic.com/
