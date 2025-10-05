# Flowchart for the Excuse My French Workflow

Below is a high‑level flowchart of the **Excuse My French** pipeline in text form.  
This chart illustrates how the components described in the accompanying documentation
fit together to generate and publish an episode featuring Butcher and his squirrel
sidekick, Nutsy. Each step flows into the next, and decision points direct the
process based on whether you have existing assets or need to generate new ones.

```
               ┌───────────────────────────┐
               │  Trending Topic Fetch  │
               │ (pytrends scheduled job)│
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │   Script Generation    │
               │    (LLM prompt with    │
               │  Butcher & Nutsy & trends)
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │     TTS Conversion     │
               │ (ElevenLabs voices for │
               │   Butcher and Nutsy)    │
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │Image Selection / Gen.  │
               │ ┌─────────────────────────┐ │
               │ │If photos exist:     │ │
               │ │  select from library│ │
               │ └─────────────────────────┘ │
               │ ┌─────────────────────────┐ │
               │ │Else:                │ │
               │ │  generate with      │ │
               │ │  DreamBooth / LoRA  │ │
               │ │  or Wan2.2 model    │ │
               │ └─────────────────────────┘ │
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │    Animation / LipSync  │
               │ (SadTalker or Wan2.2)   │
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │    Background Music    │
               │ (AI music generator    │
               │  such as Mubert)       │
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │     Video Assembly     │
               │ (ffmpeg / ComfyUI to   │
               │  combine audio, video  │
               │  and subtitles)        │
               └───────────────────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │    Publish & Monitor   │
               │ (Upload to Instagram / │
               │  TikTok; track metrics │
               │  & update trends)      │
               └───────────────────────────┘
```

### How to Read This Flowchart

1. **Trending Topic Fetch:**  
   An automated cron job uses `pytrends` to pull search and social trends.  
   Recent topics are stored in your SQLite database for later prompting.

2. **Script Generation:**  
   A call to your chosen LLM (e.g., OpenAI, Claude) produces a conversation
   between Butcher and Nutsy that blends personality traits, trending topics,
   and humour. Prompts reference the trending topic table, mood, and target
   length.

3. **TTS Conversion:**  
   The generated script is sent to ElevenLabs for speech synthesis. Choose
   distinct voices to differentiate Butcher (deep) and Nutsy (higher pitched).

4. **Image Selection / Generation:**  
   - **If you have real photos:** query your `image_library` table for
     appropriate poses or emotions and pick a matching file.  
   - **If you lack a photo:** run an image generation model. Use DreamBooth or
     LoRA trained on Butcher for still images; for dynamic video, rely on
     Wan2.2 (Wan 2 GP) which can create short video clips from prompts.

5. **Animation / LipSync:**  
   Feed each sentence and selected image into SadTalker (or Wan2.2) to produce
   animated segments with mouth movements synchronized to the audio.

6. **Background Music:**  
   Generate royalty‑free background music via an AI service like Mubert or
   SoundVerse. Keep volume low and choose moods that match the humour.

7. **Video Assembly:**  
   Use ffmpeg or ComfyUI to concatenate the lip‑synced clips, overlay
   subtitles, and mix in background music. Generate vertical videos (9:16)
   with 8 to 10 seconds length.

8. **Publish & Monitor:**  
   Manually post the video via Instagram’s or TikTok’s scheduler. Monitor
   engagement metrics (likes, shares, comments) and update your trending
   topics database for future episodes.

This textual flowchart complements the other markdown files in the `docs/`
directory and can be viewed directly on GitHub without relying on external
images.
