

import json
from typing import Any, Callable, List, Union

from tqdm import tqdm
from chatboard.text.llms.conversation import AIMessage
from chatboard.text.llms.prompt import Prompt
from chatboard.text.vectors.utils import chunks
import asyncio



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
            batch_size=10,
            retry: int=3,
            to_file: str=None
        ):
        if not self.prompt.rag_namespace:
            raise ValueError("RAG namespace is not set")
        ids = []
        keys = []
        values = []
        scores = []
        retries = []
        # for rec in dataset:
        #     hint = hint_fn(rec['label']) if hint_fn else ""
        #     try:
        #         score = None
        #         for i in range(retry):
        #             msgs = await self.prompt(**rec['inputs'], prompt_postfix=hint, output_conversation=True)
        #             if score_fn is None:
        #                 break
        #             score = score_fn(msgs[-1], rec['label'])
        #             if score:
        #                 break
        #         else:
        #             print("Failed to get a valid response")            
        #             continue
        #         scores.append(score)
        #         ids.append(rec.get("id", None))
        #         keys.append(msgs[-2].content.replace(hint, ""))
        #         values.append(msgs[-1].content)
        #         retries.append(i)
        #     except Exception as e:
        #         print(e)
        target_file = None
        if to_file:
            target_file = open(to_file, "w")

        output = []
        try:               
            for chunk in tqdm(chunks(dataset, batch_size), total=len(dataset)//batch_size):
                tasks = await asyncio.gather(*[self._transform_one(rec, score_fn, hint_fn, retry) for rec in chunk])
                output+=tasks
                if target_file:
                    for task in tasks:
                        target_file.write(json.dumps(task) + "\n")
                # ids += [task[0] for task in tasks]                        
                # keys += [task[1][-2] for task in tasks]
                # values += [task[1][-1] for task in tasks]
                # scores += [task[2] for task in tasks]
                # retries += [task[3] for task in tasks]
                # ids.append(rec.get("id", None))
                # keys.append(msgs[-2].content.replace(hint, ""))
                # values.append(msgs[-1].content)
                # retries.append(i)
        except KeyboardInterrupt as e:
            pass
        except asyncio.CancelledError as e:
            pass
        return output
        # return ids, keys, values, scores, retries


    async def _transform_one(self, rec, score_fn=None, hint_fn=None, retry=3):
        hint = hint_fn(rec['label']) if hint_fn is not None else ""
        score = None
        try:
            for i in range(retry):            
                msgs = await self.prompt(**rec['inputs'], prompt_postfix=hint, output_conversation=True)
                msgs[-2].content = msgs[-2].content.replace(hint, "")
                if score_fn is not None:
                    score = score_fn(msgs[-1], rec['label'])
                if score_fn is None or score:
                    # return rec.get("id", None), msgs, score, retry
                    return {
                        "id": rec.get("id", None),
                        "key": msgs[-2].content,
                        "value": msgs[-1].content,
                        "score": score,
                        "retry": i,
                        "error": None,
                    }
            else:
                print("Failed to get a valid response")            
                # return rec.get("id", None), msgs, score, retry
                return {
                        "id": rec.get("id", None),
                        "key": msgs[-2].content,
                        "value": msgs[-1].content,
                        "score": score,
                        "retry": i,
                        "error": None,
                    }
        except Exception as e:
            print(e)
            # return rec.get("id", None), msgs, score, retry
            return {
                "id": rec.get("id", None),
                "key": None,
                "value": None,
                "score": None,
                "retry": None,
                "error": str(e),
            }
    

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

    