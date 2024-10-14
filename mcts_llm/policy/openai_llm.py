from .base import LLMPolicy
from typing import List, Dict
from openai import OpenAI

client = OpenAI()

SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions and help with tasks.

You should make a step by step reasoning to answer the question.
- You should divide your reasoning steps by ### Step X: ...
You should make a final answer at the end of your reasoning.
- You should put the final answer in the box: ### Final Answer

For example, 

# Example 1

User : 
Question: What is sin(60)*cos(60)?
Reasoning: 
### Step 1: Recall the values of sine and cosine for \(60^\circ\)
From trigonometric tables, we know:
\[
\sin(60^\circ) = \frac{\sqrt{3}}{2}
\]
\[
\cos(60^\circ) = \frac{1}{2}
\]

Output : 
### Step 2: Multiply the values
Now, multiply the two values:
\[
\sin(60^\circ) \times \cos(60^\circ) = \left(\frac{\sqrt{3}}{2}\right) \times \left(\frac{1}{2}\right)
\]

# Example 2
User : 
Question: What is sin(60)*cos(60)?
Reasoning: 

... Skip ...

### Step 3: Simplify the expression
Multiply the numerators and denominators:
\[
\frac{\sqrt{3}}{2} \times \frac{1}{2} = \frac{\sqrt{3}}{4}
\]

Output : 
### Final Answer
So, the result of \(\sin(60^\circ) \times \cos(60^\circ)\) is:
\[
\frac{\sqrt{3}}{4}
\]


"""

STOP_WORDS = ['### step', '### final answer']

def apply_stop_words(content: str, stop_words: List[str]) -> str:
    lines = content.split('\n')
    result = []
    step_encountered = False
    for line in lines:
        if any(stop_word in line.lower() for stop_word in stop_words):
            if not step_encountered:
                result.append(line)
                step_encountered = True
            else:
                break
        else:
            result.append(line)
    refined_str = '\n'.join(result).strip()
    done = '### final answer' in refined_str.lower()
    return refined_str, done

class OpenAILLM(LLMPolicy):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def run(self, state: List[Dict[str, str]]):
        state.insert(0, {
            "role": "system",
            "content": SYSTEM_MESSAGE
        })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=state
        )

        steps = response.choices[0].message.content
        steps, done = apply_stop_words(steps, STOP_WORDS)

        return steps, done
