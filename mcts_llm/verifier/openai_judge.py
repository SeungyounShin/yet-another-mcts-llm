from typing import List, Dict
from openai import OpenAI

client = OpenAI()

SYSTEM_MESSAGE = """
You are a fair judge tasked with evaluating the quality of answers to given questions. You must rate the answer on a scale from -10 to 10.

Evaluation criteria:
1. Accuracy: Does the answer correctly address the question?
2. Completeness: Does the answer cover all aspects of the question?
3. Clarity: Is the answer clear and easy to understand?
4. Logic: Is the reasoning process in the answer logical?
5. Creativity: Does the answer provide original insights or perspectives?

Scoring guidelines:
- 10: Perfect answer. Excels in all criteria.
- 7 to 9: Excellent answer. Outstanding in most criteria.
- 4 to 6: Good answer. Above average in all criteria.
- 1 to 3: Average answer. Good in some criteria but room for improvement.
- 0: Neutral answer. Neither particularly good nor bad.
- -1 to -3: Below average answer. Needs improvement in several criteria.
- -4 to -6: Poor answer. Inadequate in most criteria.
- -7 to -9: Very poor answer. Serious issues in almost all criteria.
- -10: Completely inappropriate answer. Irrelevant or contains seriously incorrect information.

Please provide your evaluation in the following format:
Score: [Enter an integer score between -10 and 10 here]
Reason: [Brief explanation for the score]
"""

class OpenAIJudge:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def evaluate(self, messages: List[Dict[str, str]]) -> Dict[str, int]:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            *messages
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        evaluation = response.choices[0].message.content
        print(evaluation)

        # Extract score
        score_line = [line for line in evaluation.split('\n') if line.startswith('Score:')][0]
        score = int(score_line.split(':')[1].strip())

        return {"score": score}

def judge(question: str, answer: str, model_name: str = "gpt-4") -> Dict[str, int]:
    judge = OpenAIJudge(model_name)
    return judge.evaluate(question, answer)
