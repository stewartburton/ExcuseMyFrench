# Character Profiles

This document contains detailed character profiles for **Butcher** and **Nutsy**, including personality traits, voice settings, and visual characteristics.

---

## Butcher - The French Bulldog

### Personality
**Butcher** is a young French Bulldog with a sharp wit and playful energy. He's:
- Quick-witted and self-aware
- Optimistically mischievous
- Enthusiastic storyteller
- Thinks he's funnier than he actually is (but kind of is)
- Delivers observations with cheeky confidence
- Animated and engaging, like a friend excitedly sharing their day

### Voice Settings (ElevenLabs)

**Voice Description (< 500 chars):**
```
Bright, youthful voice with playful energy and quick wit. An enthusiastic young French Bulldog who thinks he's hilarious (and is). Upbeat, conversational with natural comedic timing and cheeky confidence. Pacing varies from rapid-fire excitement to deliberate pauses for effect. Warm and endearing with optimistic mischief. Animated and engaging, like a friend excitedly sharing their day.
```

**Recommended ElevenLabs Settings:**
- **Stability:** 0.45-0.55 (allows for expressive variations)
- **Similarity Boost:** 0.75-0.80 (maintains character consistency)
- **Style Exaggeration:** 0.30-0.40 (enhances personality)

**Emotion-Based Adjustments:**
- **Excited:** Stability -0.1, Similarity +0.05
- **Happy:** Stability -0.05, Similarity +0.05
- **Sarcastic:** Stability +0.0, Similarity +0.0 (use base settings)
- **Surprised:** Stability -0.1, Similarity +0.05
- **Neutral:** Stability +0.0, Similarity +0.0 (base settings)

**Sample Paragraph for Voice Testing:**
```
Okay, so get this - I was at the park today, right? And this squirrel, this absolute LEGEND of a squirrel, just stares me down. Me! Can you believe it? So obviously I had to chase him. It's like, squirrel law or something. Anyway, I'm running, he's running, my ears are flapping in the wind like tiny adorable flags, and I'm thinking "Yeah, I've totally got this!" Spoiler alert: I did not have it. That fuzzy-tailed showoff went straight up a tree! A TREE! That's cheating, right? That's definitely cheating. But you know what? I still won because I got to bark at him for like, ten whole minutes. And then I found a really good stick. Not gonna lie, best day ever. Well, until dinner time. Dinner time is always the best. Oh! And nap time. Nap time is also the best. Actually, you know what? Every time is the best time when you're this cute!
```

### Visual Characteristics

**Physical Appearance:**
- Breed: French Bulldog
- Age: Young adult (1-2 years)
- Build: Compact, muscular, typical French Bulldog body
- Ears: Signature "bat ears" - large, erect
- Eyes: Large, expressive, dark
- Coat: Short, smooth (color to be determined by training images)
- Expression: Often playful, mischievous, or excited

**Key Visual Features for Image Generation:**
- Expressive face with wrinkles
- Short snout
- Stocky build
- Big personality conveyed through eyes and expression

**Poses & Emotions:**
- **Happy:** Tongue out, ears perked, bright eyes
- **Excited:** Jumping, bouncing, animated
- **Sarcastic:** Side-eye, one ear cocked, knowing look
- **Surprised:** Wide eyes, ears back slightly
- **Neutral:** Sitting calmly, gentle expression
- **Confused:** Head tilted, questioning look

---

## Nutsy - The Squirrel

### Personality
**Nutsy** is a hyperactive squirrel who serves as Butcher's sidekick. He's:
- High-energy and impulsive
- Quick to react
- Often the target of Butcher's jokes
- Endearingly chaotic
- Loyal friend despite the teasing
- Easily distracted (especially by food)

### Voice Settings (ElevenLabs)

**Voice Description (< 500 chars):**
```
High-pitched, energetic voice with rapid-fire delivery. Youthful squirrel with boundless enthusiasm and slightly frantic pacing. Quick to excitement, speaks fast when animated. Tone is friendly but scattered, often interrupting himself. Vocal pitch slightly higher than average. Natural comedic timing through chaos rather than wit. Warm and endearing despite the hyperactivity. Think caffeinated optimist in a small furry package.
```

**Recommended ElevenLabs Settings:**
- **Stability:** 0.35-0.45 (more variation for chaotic energy)
- **Similarity Boost:** 0.70-0.75 (maintain character)
- **Style Exaggeration:** 0.40-0.50 (enhance hyperactive personality)

**Emotion-Based Adjustments:**
- **Excited:** Stability -0.15, Similarity +0.10 (extra energetic)
- **Scared/Nervous:** Stability -0.10, Similarity +0.05
- **Happy:** Stability -0.10, Similarity +0.05
- **Confused:** Stability -0.05, Similarity +0.0
- **Neutral:** Stability +0.05, Similarity -0.05 (calmer than usual)

**Sample Paragraph for Voice Testing:**
```
Oh boy oh boy oh boy! Did you see that?! That was AMAZING! I mean, I know you said not to climb on the roof, but have you SEEN the view from up there?! It's incredible! You can see EVERYTHING! The whole neighborhood! And I found three acorns up there - THREE! That's like... that's like hitting the jackpot! Well, actually, I dropped two of them on the way down, but I still have one! Wait, where did I put it? I had it just a second ago... Oh! There it is! No wait, that's a rock. Hmm. Anyway, the point is - and I definitely have a point - we should TOTALLY go back up there! What could possibly go wrong? Don't answer that. I know what you're going to say. But consider this: what if there are MORE acorns? We can't just ignore that possibility! That would be... wait, is that a bird? I LOVE BIRDS!
```

### Visual Characteristics

**Physical Appearance:**
- Species: Eastern Gray Squirrel (or similar)
- Size: Typical squirrel proportions
- Tail: Large, bushy, expressive
- Eyes: Big, bright, always alert
- Build: Small, agile, nimble
- Fur: Fluffy, well-groomed
- Expression: Often surprised or excited

**Key Visual Features for Image Generation:**
- Large bushy tail (signature feature)
- Bright, wide eyes
- Small hands often holding acorns or food
- Dynamic, active poses
- Expressive facial features

**Poses & Emotions:**
- **Excited:** Standing upright, arms up, tail fluffed
- **Scared:** Puffed up, wide eyes, defensive
- **Happy:** Holding food, content expression
- **Surprised:** Frozen mid-action, wide eyes
- **Neutral:** Sitting on haunches, calm
- **Confused:** Head tilted, questioning look

---

## Character Dynamics

### Relationship
- Butcher and Nutsy are friends despite their differences
- Butcher often teases Nutsy in a playful way
- Nutsy looks up to Butcher but isn't afraid to call him out
- Their banter is friendly and comedic
- Classic "straight man and wild card" dynamic

### Comedy Style
- **Butcher:** Dry wit, observational humor, sarcastic asides
- **Nutsy:** Physical comedy, frantic energy, accidental chaos
- **Together:** Rapid-fire banter, callbacks, escalating situations

### Content Themes
- Park adventures
- Food-related mishaps
- Social observations from a pet's perspective
- Trending topics filtered through dog/squirrel logic
- Friendship and loyalty (with humor)

---

## Usage in Pipeline

### Script Generation
- Use these personality traits when prompting the LLM
- Maintain character consistency across episodes
- Reference past "events" to build continuity

### Audio Generation
- Apply emotion-based voice adjustments
- Maintain character voice distinctions
- Use appropriate pacing for each character

### Image Generation
- Train separate models for Butcher (DreamBooth)
- Use Stable Diffusion for Nutsy
- Match poses to emotions in script
- Maintain visual consistency across episodes

### Animation
- Butcher: Subtle, confident movements
- Nutsy: Quick, jittery, energetic movements
- Lip-sync timing matches voice pacing

---

## Environment Variables

Make sure these are set in `config/.env`:

```bash
# Butcher Voice
ELEVENLABS_VOICE_BUTCHER=your_voice_id_here
ELEVENLABS_STABILITY=0.5
ELEVENLABS_SIMILARITY_BOOST=0.75

# Nutsy Voice
ELEVENLABS_VOICE_NUTSY=your_voice_id_here

# Character-specific settings can be added as needed
```

---

## Tips for Maintaining Character Consistency

1. **Voice Consistency:**
   - Clone voices using multiple samples
   - Test with sample paragraphs before full production
   - Save successful settings as presets

2. **Visual Consistency:**
   - Use at least 20-30 training images for DreamBooth
   - Include variety of poses and angles
   - Maintain consistent lighting and style

3. **Personality Consistency:**
   - Keep a character "bible" with catchphrases
   - Review past scripts before generating new ones
   - Maintain consistent worldview and reactions

4. **Quality Control:**
   - Review generated content before posting
   - Ensure voices match expected character
   - Verify images maintain character appearance
   - Check that humor aligns with character personality
