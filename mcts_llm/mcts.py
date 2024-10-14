from mcts_llm.policy import LLMPolicy, OpenAILLM
from mcts_llm.verifier import LLMVerifier, OpenAIJudge
from typing import List, Dict, Literal
import numpy as np
import math
import os

class Node:
    def __init__(self, question: str, x : List[str] = []):
        self.question = question
        self.x: List[str] = x
        
        self.N: int = 0
        self.Q: float = 0.0

        self.parent: Node = None
        self.children: List[Node] = []
    
    def __repr__(self):
        _str = f"Question: [{self.question}]\n"
        for i, step in enumerate(self.x):
            _str += "-" * 10
            _str += f"\n{step}\n"
        _str += "-" * 10
        return _str
    
    def select_child(self, c=math.sqrt(2)):
        # UCB1 formula for MCTS
        total_visits = sum(child.N for child in self.children)
        
        ucb_values = [
            (child.Q / child.N) + c * math.sqrt(math.log(total_visits) / child.N)
            if child.N > 0 else float('inf')
            for child in self.children
        ]

        return self.children[np.argmax(ucb_values)]

    def _make_message(self, mode : Literal['policy', 'verifier']) -> List[Dict[str, str]]:
        question = self.question
        steps = []
        for i, step in enumerate(self.x):
            cleaned_step = step.replace(f'### Step {i+1}:', '').strip()
            steps.append(f'### Step {i+1}:\n{cleaned_step}')
        prefix_reasoning = '\n'.join(steps)

        if mode == 'policy':
            message = f"""
            Question:\n{question}
            Reasoning:\n{prefix_reasoning}

            Continue reasoning or provide the final answer.
            """
        elif mode == 'verifier':
            message = f"""
            Question:\n{question}
            Reasoning and answer:\n{prefix_reasoning}
            """

        messages = [
            {
                "role": "user",
                "content": message
            }
        ]

        return messages     


class MCTS:
    def __init__(self, question: str, policy: LLMPolicy, verifier: LLMVerifier, iteration: int):
        self.question = question
        self.policy = policy
        self.verifier = verifier
        self.root = Node(
            question=question
        )

        self.iteration = iteration
        self.max_depth = iteration//8
        self.max_children = 3

    def _backpropagate(self, node: Node, score: float):
        while node:
            node.N += 1
            node.Q += score
            node = node.parent

    def _visualize(self, prefix: str = ""):
        from graphviz import Digraph
        dot = Digraph(comment='MCTS Tree')

        def add_nodes_edges(node):
            node_id = str(id(node))

            # Determine the label for the node
            if len(node.x) == 0:
                # Root node, display the question in bold and slightly larger font, left aligned
                label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" ALIGN="LEFT"><TR><TD ALIGN="LEFT"><B>Q={node.Q:.2f}</B><BR/>N={node.N}<BR/><I>Question:</I><BR/><I>{node.question}</I></TD></TR></TABLE>>'
                dot.node(node_id, label=label, shape='ellipse', style='filled', color='lightblue', fontname="Arial", fontsize="12")
            else:
                # Non-root node, display the latest action with left alignment and proper line breaks
                latest_action = node.x[-1].replace('\n', '<BR ALIGN="LEFT"/>')  # Replace newlines with <BR/>
                label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" ALIGN="LEFT"><TR><TD ALIGN="LEFT"><B>Q={node.Q:.2f}</B><BR/>N={node.N}<BR/><I>Action:</I><BR/><I>{latest_action}</I></TD></TR></TABLE>>'
                dot.node(node_id, label=label, shape='box', style='filled', color='lightyellow', fontname="Arial", fontsize="10")

            # Recursively add children
            for child in node.children:
                child_id = str(id(child))
                add_nodes_edges(child)
                dot.edge(node_id, child_id, color='gray', arrowhead='vee')

        # Build the graph starting from the root node
        add_nodes_edges(self.root)

        # Render the graph to a PNG file
        dot.format = 'png'
        if not os.path.exists('./visualization'):
            os.makedirs('./visualization')
        output_path = f'./visualization/{prefix}'
        dot.render(output_path, view=False)
        print(f"Tree visualization saved as '{output_path}.png'")


    def run(self):

        current_node = self.root

        for _ in range(self.iteration):
            self._visualize(prefix=f"iteration_{_}")

            if self.max_children > len(current_node.children):
                # Expand
                current_message = current_node._make_message(mode='policy')
                action, done = self.policy.run(
                    state=current_message
                )
                child_node = Node(
                    question=self.question,
                    x=current_node.x + [action]
                )
                child_node.parent = current_node
                current_node.children.append(child_node)
            else:
                if len(current_node.children) == 0 and len(current_node.x) > 0:
                    done = True
                else:
                    done = False 

            current_node = current_node.select_child()

            if done or len(current_node.x) >= self.max_depth:
                print(current_node)
                # Verify
                score = self.verifier.evaluate(
                    messages=current_node._make_message(mode='verifier')
                )
                print(f"Score: {score}")
                # Backpropagate
                self._backpropagate(node=current_node, score=score['score'])
                current_node = self.root
            
        

if __name__ == "__main__":

    question = "how many 'r's in starberry?"
    policy = OpenAILLM(
        model_name="gpt-4o-mini"
    )
    verifier = OpenAIJudge(
        model_name="gpt-4o"
    )

    mcts = MCTS(
        question=question,
        policy=policy,
        verifier=verifier,
        iteration=32
    )
    
    mcts.run()
