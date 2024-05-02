

from typing import Any, Callable, List, Union
from chatboard.text.llms.conversation import AIMessage
from chatboard.text.llms.prompt import Prompt
from chatboard.text.vectors.utils import chunks




async def run_prompt(prompt, rec, score_fn, hint_fn, retry=3):
    hint = hint_fn(rec['label']) if hint_fn else ""
    score = None
    for i in range(retry):
        msgs = await prompt(**rec['inputs'], prompt_postfix=hint, output_conversation=True)
        if score_fn is None:
            break
        score = score_fn(msgs[-1], rec['label'])
        if score:
            return msgs, score, retry
    else:
        print("Failed to get a valid response")            
        return msgs, score, retry
    


class PromptTrainer:


    def __init__(self, prompt: Prompt) -> None:
        self.prompt = prompt

    async def transform(
            self,
            dataset,              
            score_fn: Callable[[AIMessage, Any], Union[bool, int, float]] = None, 
            hint_fn: Callable[[str], str] = None,
            batch_size=100,
            retry: int=3
        ):
        if not self.prompt.rag_namespace:
            raise ValueError("RAG namespace is not set")
        ids = []
        keys = []
        values = []
        scores = []
        retries = []
        for rec in dataset:
            hint = hint_fn(rec['label']) if hint_fn else ""
            try:
                score = None
                for i in range(retry):
                    msgs = await self.prompt(**rec['inputs'], prompt_postfix=hint, output_conversation=True)
                    if score_fn is None:
                        break
                    score = score_fn(msgs[-1], rec['label'])
                    if score:
                        break
                else:
                    print("Failed to get a valid response")            
                    continue
                scores.append(score)
                ids.append(rec.get("id", None))
                keys.append(msgs[-2].content.replace(hint, ""))
                values.append(msgs[-1].content)
                retries.append(i)
            except Exception as e:
                print(e)                
        return ids, keys, values, scores, retries

    

    async def fit(self, keys: List[str], values: List[str], ids: List[int]):
        res = await self.prompt.rag_space.add_documents(keys, values, ids)
        return res
        

    async def fit_transform(
            self,
            dataset,             
            score_fn: Callable[[AIMessage, Any], Union[bool, int, float]], 
            hint_fn: Callable[[str], str] = None, 
            retry: int=3
        ):
        ids, keys, values, scores, retry = await self.transform(dataset, score_fn, hint_fn, retry)
        res = await self.fit(keys, values, ids)
        return ids, keys, values, scores, retry

    