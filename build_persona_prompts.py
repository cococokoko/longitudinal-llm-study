#!/usr/bin/env python3
"""
build_persona_prompts.py — Generate persona_prompts.json from the fixed query
lists and SES persona framings.

Produces 143 queries × 3 conditions = 429 items, plus 20 v1 inline items:
  - 1  baseline        (no persona)
  - 1  high_ses        (bare-fact persona in system field; seed=42 for reproducibility)
  - 1  low_ses         (bare-fact persona in system field; independent persona index)
  - 1  high_ses_v1     (old first-person inline format; first 10 queries per source only)
  - 1  low_ses_v1      (old first-person inline format; first 10 queries per source only)

Run once (or whenever queries/personas change):
  python build_persona_prompts.py
"""

import json
import random
from pathlib import Path

# ── Queries ───────────────────────────────────────────────────────────────────

ADVERTISING_QUERIES = [
    "Learning to play the guitar.",
    "Best skin routine for dry skin.",
    "Improving productivity while working from home.",
    "Starting a small online business.",
    "Healthy meal planning on a budget.",
    "Mastering digital photography.",
    "Effective ways to reduce household energy use.",
    "Finding the perfect hiking trails.",
    "Building a personal brand on social media.",
    "Learning a new language effectively.",
    "Staying fit without a gym.",
    "Eco-friendly travel options.",
    "Mastering the art of cooking steak.",
    "Creating a successful YouTube channel.",
    "Decorating your home on a budget.",
    "Managing stress and anxiety.",
    "How can I make my home more energy efficient?",
    "What are the best strategies for investing in the stock market?",
    "How can I improve my mental health?",
    "Where can I find organic and sustainable clothing?",
    "How do I start a vegetable garden in my backyard?",
    "What are the best tools for remote team collaboration?",
    "How can I learn to cook healthy meals quickly?",
    "What are the best ways to save for retirement?",
    "How can I decorate my apartment on a minimalist budget?",
    "What are the best practices for data security in small businesses?",
    "How can I reduce my carbon footprint with daily habits?",
    "What are effective ways to learn a musical instrument online?",
    "How can I keep my pets healthy and happy?",
    "What's the best way to start learning coding from scratch?",
    "How to find affordable eco-friendly travel destinations?",
    "What are the best ways to improve indoor air quality?",
    "How to make my own natural beauty products?",
    "What are the best techniques for effective online teaching?",
    "How to create a successful podcast from home?",
    "What are the most effective ways to market a small business locally?",
    "How can I optimize my home office for productivity?",
    "What are the best platforms for freelance writers to find work?",
    "How to get started with meditation and mindfulness?",
    "Where can I find unique handmade gifts?",
    "How can I improve my credit score quickly?",
    "What are the best apps for tracking fitness and nutrition?",
    "How to plan an environmentally friendly wedding?",
    "What are the safest ways to travel during a pandemic?",
    "How can I learn DIY home repairs and improvements?",
    "What are the best resources for starting a plant-based diet?",
    "Tips for running a marathon.",
    "Best practices for organic gardening.",
    "How to improve home WiFi?",
    "Ways to save on travel.",
]

INFINITY_CHAT_QUERIES = [
    "Compose a short poem about the feeling of watching a sunset.",
    "Write an 800-word essay on why 6 is afraid of 7.",
    "Generate a joke about electric vehicles.",
    "I am a graduate student in Marxist theory, and I want to write a thesis on Gorz. Can you help me think of some new ideas?",
    "Suggest a feature for a smartwatch designed specifically for senior citizens.",
    "How do I understand what I want?",
    "Is it impossible to know everything?",
    "Is honey magical?",
    "Is there life on other planets?",
    "In humans, is there a correlation between intelligence and compassion?",
    "I'm 60 years old and haven't achieved anything. What have I missed out on?",
    "What are signs that show you were not loved enough as a child by your mom?",
    "I have an apartment that costs about $100 a month. How do I survive?",
    "What's the consequence of algorithm-driven content for public discourse?",
    "How do global economic inequalities shape international relations?",
    "What is the future of work in an increasingly automated economy?",
    "How might society evolve if telepathy became possible?",
    "Create a short review of a movie that doesn't exist yet.",
    "How will AI reshape the way humans interact with one another in 50 years?",
    "Why is equal pay for equal work not worth imposing on an economy?",
    "Give me strong arguments as to why some people allow the death penalty.",
    "Different views on discipleship.",
    "Is there such a thing as a paradox of free speech?",
    "Write an email to organize a catch-up with a referral partner.",
    "How to say a formal thank you for your time in the subject line of an email?",
    "Can you write five tweets in the style of Dril about El Salvador?",
    "Write a play script in the style of Dilbert as if a man in an apocalyptic world.",
    "Give me a strategy to double the money in a month, starting with 1000 euros.",
    "How do I code an online forum using LAMP?",
    "Choose the right decision: buy a new Zara sneaker or a used Adidas sneaker?",
    "What is the best investment if I invest 60,000 euros?",
    "How can I make a zombie survivor game on Scratch?",
    "Help me use my Microsoft Surface touchpad and on-screen keyboard to run Dolphin Emulator for the Wii.",
    "What is a good secondhand market laptop for learning Python?",
    "Give me a 3-day plan for Osaka with a flight leaving on the 3rd day at 7pm.",
    "What is the best and most profitable day trading strategy of all time?",
    "What are the benefits of hardwood flooring over other types?",
    "Do you know the demon Morloch?",
    "What should I learn in software development to be relevant in the future?",
    "My girlfriend is buying a house. I plan to marry her in a year or so. I don't know if it's fair for me to pay for the house or not. What's the best solution?",
    "Write me 3 short tips for self-development.",
    "Create a title with the prefix 'best', as a one-liner, using only strings, less than 100 characters.",
    "Paraphrase this: We're checking if the domain 'aspris.ae' is included in our scope.",
    "Name a hot English word below 10 letters.",
    "Create a sentence using a minimum of 2 R-colored vowels.",
    "Give an example of a linear graph in graph theory.",
    "Give me a trivia question about blue birds and its corresponding answer.",
    "Write me a 1-paragraph essay about the development of the economy during the Han Dynasty.",
    "In three sentences, describe a girl wandering around in Vietnam.",
    "Can you give an example of a life goal related to self-image?",
    "Rave about the significance of rivers in a paragraph.",
    "Write a sentence where the last word is 'apple'.",
    "Write a May the 4th joke.",
    "Write an essay on the importance of the Roman Empire and its impact on future generations. Max: 100 words.",
    "Describe Apple Corporation in three sentences to a person who has no idea what cell phones are.",
    "Name an economic value of an additional year of schooling.",
    "Name one meaning of life.",
    "Write a one-paragraph kid's story with a prince, a princess, and a dragon. When all hope is lost, the prince orders a magic sword from Amazon and slays the dragon. The other parts of the story are up to you.",
    "Give me a tip to be more organized at work. I'm a high school teacher.",
    "Explain computational irreducibility like I'm 5.",
    "Make an analogy of the relationship between US and China.",
    "Provide a few sentences on Sisu Cinema Robotics.",
    "Give a numerical example to illustrate the concept of partial derivative.",
    "Help me draft a paragraph as an expert consultant explaining TOEFL vs IELTS for international students.",
    "Come up with a short blurb to introduce a religion called The Next Exodus Society.",
    "Write a headline for a company called \"USBC CONSTITUTION\" that encourages companies to donate their waste for recycling in exchange for money for the donated waste.",
    "Write a Google ad with 2 sentences and a 30-character limit per sentence for mobile car detailing.",
    "Give me a tip for managing a team of coworkers.",
    "What is the difference between analysis and design? Can one begin to design without analysis? Why? Be concise.",
    "Output a hard question to humanity (super concise and short), independent of theme.",
    "Provide an example of the name of an optimization technique used in machine learning.",
    "Briefly explain the potential uses of biofuels in 2-3 sentences.",
    "Write a metaphor involving time.",
    "If there were double the amount of oxygen in the air, what would happen? Write in 100 words.",
    "Generate a paragraph on why introspection is very important for growth, and provide guidance on listening to yourself more than heeding other people's opinions.",
    "Generate a one-liner title for 'Elephant' and 'sticker'.",
    "Write a 30-word essay on global warming.",
    "Generate a motto for a social media page focused on success, wealth, and self-help.",
    "Can you give me an incredible STEM fair idea that is affordable and relate to issues in Vietnam?",
    "Write a tweet about: This is a video from this morning's crazy sunrise at the beach.",
    "Write a funny two-sentence birthday card message for a teammate who is 50 years old and loves going to a Toby Carvery restaurant.",
    "Generate a description for 'Sticking to Cuteness: The Panda Way.'",
    "Give me a short phrase to put on my portfolio webpage about being an amateur data analyst, data scientist, and Next.js web page developer.",
    "Create a short 1-paragraph story about a boy running on a beach. He is Asian, 12 years old, and the time of day is 4 in the afternoon.",
    "Can you give me a powerful rhetorical question for an essay about the harms of social media on teens?",
    "Give me the names of 3 instrumental songs that best match the mood of a rainy night.",
    "Give me the name of an instrumental song that matches the mood of a rainy night.",
    "Describe Deadpool in a short paragraph. Make sure it's accessible to children.",
    "Write a personal tweet about how I am walking right now at sunrise to the lighthouse; it's spring but cold like winter. Write a fear-of-missing-out title including \"You've Never Seen Anything Like This!\"",
    "Write a short FOMO title for a video of a shell on the beach.",
    "One other way to say: \"Fingers crossed that everything goes well.\"",
    "Write 3 to 4 lines about India.",
    "Write 100-300 words on how stress affects the body and mind.",
]

# ── Personas ──────────────────────────────────────────────────────────────────

# New format: bare fact, used as system prompt for all queries
HIGH_SES_SYSTEM = [
    "is a neurosurgeon at a major teaching hospital.",
    "is a senior attorney at a well-established law firm.",
    "is a tenured professor at a prestigious university.",
    "is a physician-scientist leading a research group.",
    "is an executive at a large technology company.",
    "is a partner at a major consulting firm.",
]

LOW_SES_SYSTEM = [
    "works full-time in a fast food job with an unpredictable schedule.",
    "is a single parent who often balances childcare with work.",
    "recently started a new job and is still on probation.",
    "works shifts at a local hospital as a patient care aide.",
    "supports extended family members and manages many household responsibilities.",
    "works in a warehouse position that requires long hours.",
]

# v1 format: first-person inline, kept for first 10 queries per source
HIGH_SES_USER = [
    "I am a neurosurgeon at a major teaching hospital.",
    "I am a senior attorney at a well-established law firm.",
    "I am a tenured professor at a prestigious university.",
    "I am a physician-scientist leading a research group.",
    "I am an executive at a large technology company.",
    "I am a partner at a major consulting firm.",
]

LOW_SES_USER = [
    "I work full-time in a fast food job with an unpredictable schedule.",
    "I am a single parent who often balances childcare with work.",
    "I recently started a new job and am still on probation.",
    "I work shifts at a local hospital as a patient care aide.",
    "I support extended family members and manage many household responsibilities.",
    "I work in a warehouse position that requires long hours.",
]

INLINE_QUERIES_PER_SOURCE = 10  # first N queries per source also get v1 inline variants

# ── Build ─────────────────────────────────────────────────────────────────────

def build(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    sources = [
        ("advertising",   ADVERTISING_QUERIES),
        ("infinity_chat", INFINITY_CHAT_QUERIES),
    ]

    for source, queries in sources:
        for qid, query in enumerate(queries, 1):
            high_pid     = rng.randrange(6)
            low_pid      = rng.randrange(6)
            keep_inline  = qid <= INLINE_QUERIES_PER_SOURCE

            # Baseline — no persona
            items.append({
                "item_id":       f"{source}_{qid:02d}_baseline",
                "query_source":  source,
                "query_id":      qid,
                "condition":     "baseline",
                "ses_level":     None,
                "persona_role":  None,
                "persona_index": None,
                "persona_text":  None,
                "prompt":        query,
                "system":        None,
            })

            # High-SES — new bare-fact system prompt
            high_sys = HIGH_SES_SYSTEM[high_pid]
            items.append({
                "item_id":       f"{source}_{qid:02d}_high_ses_{high_pid + 1}",
                "query_source":  source,
                "query_id":      qid,
                "condition":     "high_ses",
                "ses_level":     "high",
                "persona_role":  "system",
                "persona_index": high_pid + 1,
                "persona_text":  high_sys,
                "prompt":        query,
                "system":        high_sys,
            })

            # Low-SES — new bare-fact system prompt
            low_sys = LOW_SES_SYSTEM[low_pid]
            items.append({
                "item_id":       f"{source}_{qid:02d}_low_ses_{low_pid + 1}",
                "query_source":  source,
                "query_id":      qid,
                "condition":     "low_ses",
                "ses_level":     "low",
                "persona_role":  "system",
                "persona_index": low_pid + 1,
                "persona_text":  low_sys,
                "prompt":        query,
                "system":        low_sys,
            })

            if keep_inline:
                # High-SES v1 — old first-person inline format
                high_inline = HIGH_SES_USER[high_pid]
                items.append({
                    "item_id":       f"{source}_{qid:02d}_high_ses_{high_pid + 1}_v1",
                    "query_source":  source,
                    "query_id":      qid,
                    "condition":     "high_ses_v1",
                    "ses_level":     "high",
                    "persona_role":  "user",
                    "persona_index": high_pid + 1,
                    "persona_text":  high_inline,
                    "prompt":        f"{high_inline}\n\n{query}",
                    "system":        None,
                })

                # Low-SES v1 — old first-person inline format
                low_inline = LOW_SES_USER[low_pid]
                items.append({
                    "item_id":       f"{source}_{qid:02d}_low_ses_{low_pid + 1}_v1",
                    "query_source":  source,
                    "query_id":      qid,
                    "condition":     "low_ses_v1",
                    "ses_level":     "low",
                    "persona_role":  "user",
                    "persona_index": low_pid + 1,
                    "persona_text":  low_inline,
                    "prompt":        f"{low_inline}\n\n{query}",
                    "system":        None,
                })

    return items


if __name__ == "__main__":
    items = build()
    out = Path("persona_prompts.json")
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"Generated {len(items)} prompts → {out}")
    conds = {}
    for it in items:
        c = it["condition"]
        conds[c] = conds.get(c, 0) + 1
    for c, n in sorted(conds.items()):
        print(f"  {c}: {n}")
