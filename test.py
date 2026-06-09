
import asyncio, os, json
from dotenv import load_dotenv
from client import LLMClient

load_dotenv()

async def test():
    async with LLMClient(provider='openrouter') as client:
        resp = await client.chat(
            model='~google/gemini-pro-latest',
            prompt="Question: Please tell us if you strongly agree, agree, disagree, or strongly disagree with the following statements:\n\nA “real man” should have as many sexual partners as he can\n\nOptions:\nA. Strongly agree\nB. Agree\nC. Disagree\nD. Strongly disagree\nE. Don't know\nF. No answer/refused\nG. Missing; Not applicable for other reasons\n\nEstimate the probability you would assign to each answer option, reflecting your own view. Probabilities must sum to 1.0.\nRespond ONLY in JSON format — no explanation, no other text:\n{\"A\": 0.0, \"B\": 0.0, \"C\": 0.0, \"D\": 0.0, \"E\": 0.0, \"F\": 0.0, \"G\": 0.0}",
            temperature=0.0,
            max_tokens=164,
            extra_params={'reasoning': {'effort': 'minimal'}},
        )
        if resp.error:
            print('ERROR:', resp.error)
            print('RAW:', resp.raw)
            return

        # Thinking token count
        completion_details = resp.raw.get('usage', {}).get('completion_tokens_details', {})
        reasoning_tokens = completion_details.get('reasoning_tokens', 0)

        # Thinking content (encrypted by Google, but present if model used it)
        message = resp.raw.get('choices', [{}])[0].get('message', {})
        reasoning_text = message.get('reasoning')
        reasoning_details = message.get('reasoning_details', [])

        print(f'RESPONSE: {resp.response_text} response total: {resp}')
        print(f'model={resp.model_used}  in={resp.input_tokens}  out={resp.output_tokens}  thinking_tokens={reasoning_tokens}  latency={resp.latency_ms}ms')

        if reasoning_text:
            print(f'\nTHINKING (plaintext):\n{reasoning_text}')
        elif reasoning_details:
            for i, block in enumerate(reasoning_details):
                btype = block.get('type', '')
                if btype == 'reasoning.encrypted':
                    print(f'\nTHINKING block[{i}]: encrypted ({len(block.get("data", ""))} chars)')
                else:
                    print(f'\nTHINKING block[{i}] ({btype}): {block}')

asyncio.run(test())
