from langsmith import RunTree
import numpy as np
from pydantic import BaseModel
from components.etl.chat_prompt import ChatPrompt, ChatResponse
from components.etl.completion_parsing import num_split_field
from components.etl.system_conversation import AIMessage, Conversation, ConversationRag
from components.etl.rag_manager import RagVectorSpace
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool

from components.etl.tracer import Tracer









class FinishAction(BaseModel):
    """a finish action to indicate you are satisfied with the result that there is nothing else to do and all the frames are ready."""
    to_finish: bool = Field(..., description="a boolean to indicate that the action is finished.")

class FinishActionException(Exception):
    pass



class ShortTermMessageMemory:

    def __init__(self, memory_length=5):
        self.conversation = []
        self.memory_length = memory_length

    def add_message(self, message):
        self.memory.append(message)

    def get_memory(self):
        return self.memory[-self.memory_length:]        



class LongTermRagMessageMemory:
    pass



def get_tool_scheduler_prompt(tool_dict):
    tool_function = tool_dict['function']
    prompt = f"""{tool_function["name"]}: {tool_function["description"]}\n\tparameters:"""
    for prop, value in tool_function["parameters"]['properties'].items():
        prompt += f"\n\t\t{prop}: {value['description']}"
    return prompt


# https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

class ChatAgent:

    def __init__(
            self, 
            promptpath=None,
            tools=[], 
            name="ChatAgent",
            system_filename=None, 
            user_filename=None, 
            save_rag=False, 
            rag_index=None,
            rag_length=0,
            model=None,
            stop_sequences=None,
            logit_bias=None,
        ):
        self.tools = tools + [FinishAction]
        if not name:
            raise ValueError("ChatAgent must have a name")
        self.name = name
        
        self.system_filename = system_filename
        self.user_filename = user_filename
        self.model = model
        self.prompt = ChatPrompt(
            promptpath=promptpath,
            name=name,
            system_filename=system_filename,
            user_filename=user_filename,
            model=self.model,
            stop_sequences=stop_sequences,
            logit_bias=logit_bias
        )
        self.save_rag = save_rag  
        self.conversation = Conversation()
        self.rag_index = rag_index
        self.rag_length = rag_length
        self.rag_space = None
        if rag_index:
            self.rag_space = ConversationRag(rag_index)

    
    async def next_step(self, prompt=None, step_index=None, tracer_run=None, **kwargs): 


        conversation = await self.prompt.build_conversation(
            prompt=prompt,
            conversation=self.conversation,            
            tools=[
                get_tool_scheduler_prompt(convert_to_openai_tool(tool))
            for tool in self.tools],
            tool_names=[tool.__name__ for tool in self.tools],
        **kwargs)

        with Tracer(
            tracer_run=tracer_run,
            name=f"{self.name}Step{str(step_index) if step_index is not None else ''}",
            run_type="prompt",
            inputs={
                "kwargs": kwargs,
                # "messages": conversation.messages,
            }
        ) as step_run:

            examples = None
            if self.rag_length:
                examples = await self.rag_space.similarity(conversation, self.rag_length)

            openai_messages = await self.prompt.to_openai(conversation, examples=examples)
            
            llm_response = await self.prompt.llm.send(            
                openai_messages=openai_messages,
                tracer_run=step_run,
                **kwargs,
            )

            llm_output = llm_response.choices[0].message

            parsed_response = await self.prompt.parse_completion(
                output=llm_output,
                **kwargs
            )

            ai_message = AIMessage(content=llm_output.content)
            conversation.append(ai_message)

            llm_tool_completion, tool_calls = await self.get_tool(llm_output, conversation, tracer_run=step_run)

            if tool_calls:
                conversation[-1].tool_calls = tool_calls
            self.conversation = conversation.copy()

            step_run.end(outputs={
                "messages": conversation.messages[-1],
                "funcations": llm_tool_completion,
            })

            llm_response.choices[0].message

            return ChatResponse(
                value=parsed_response,
                run_id=str(step_run.id),
                conversation=conversation,
                tools=tool_calls
            )

    async def get_tool(self, llm_output, conversation, tracer_run=None):
        tool_content = await self.parcer(llm_output.content)
        # tool_conv = Conversation([conversation[0]])
        tool_conv = Conversation()
        tool_conv.append(
            AIMessage(
                content=tool_content
            )
        )
        tool_choice = None
        # for tool in self.tools:
        #     if tool.__name__ in tool_conv.messages[-1].content:
        #         tool_choice = tool.__name__
        #         break        

        llm_tool_completion, tool_calls = await self.prompt.call_llm_with_tools(
            # conversation=conversation,
            conversation=tool_conv,
            tools=self.tools,
            tool_choice=tool_choice,
            tracer_run=tracer_run
        )
        return llm_tool_completion, tool_calls


    async def parcer(self, completion, context = None):
        # split_res = split_field("Thought", completion)
        # if len(split_res) > 1:
        #     return split_res[1]
        split_res = num_split_field("Action", completion, maxsplit=1)
        if split_res is not None and len(split_res) > 1:
            return "Action:" + split_res[1]
        return completion
    
    
    async def reducer(self, context, state, action: BaseModel = None, run_manager=None):
        # if action.type == "StockMedia":
        return state
    








class HistoryStep:

    def __init__(self) -> None:
        pass




async def run_agent(
        promptpath,
        tools,
        reducer,
        state_model,
        context,
        name="Agent",
        system_filename=None,
        user_filename=None,        
        max_iterations=10,
        rag_index=None,
        rag_length=0,
        tracer_run=None,     
        **kwargs
    ):


    with Tracer(
        tracer_run=tracer_run,
        name=name,
        run_type="chain",
        inputs={
            "kwargs": kwargs,
        }
    ) as agent_run:

    
        agent = ChatAgent(
                promptpath,
                name=name,
                # logit_bias={
                #     # "```": -30,
                #     "Thought": 5,
                #     "StateObservation": 8,
                # },
                tools=tools,
                system_filename=system_filename,
                user_filename=user_filename,
                rag_index=rag_index,
                rag_length=rag_length,
            )
        
        # state = context.store.get_init_state(str(agent_run.id), **kwargs)
        state = state_model(**kwargs)
        # state_history = [state.copy()]
        try:
            for i in range(max_iterations):
                print(f"iteration: {i}")            
                agent_output = await agent.next_step(
                    state=state, 
                    step_index=i,
                    tracer_run=agent_run, 
                    **kwargs)            
                # context.store.push(str(agent_run.id), state, agent_output.value, agent_output.tools)
                for action in agent_output.tools:
                    print("Tool:", action)
                    if type(action) == FinishAction:
                        raise FinishActionException()
                    
                    with agent_run.create_child(
                        name=type(action).__name__,
                        run_type="tool",
                        inputs={
                            "tool": action,
                        }
                    ) as tool_run:
                        state = await reducer(context, state, action, tracer_run=tool_run)
                        print("State:", state)
                        tool_run.end(outputs={
                            "state": state,
                        })
            else:
                did_stop = False
        except FinishActionException:
            did_stop = True


        agent_run.end(outputs={
            "state": state,
            "did_stop": did_stop,
        })
        return state


