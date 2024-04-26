from components.etl.prompt_tracer import PromptTracer
from langchain_core.messages import HumanMessage, AIMessage
from IPython.display import display, HTML
from langchain_openai import ChatOpenAI




class ConversationDebug:

    def __init__(self, messages, model=None):
        self.orig_messages = messages
        self.messages = [m.copy() for m in messages]
        self.llm = ChatOpenAI(model=model or "gpt-3.5-turbo")

    
    def reset(self):
        self.messages = [m.copy() for m in self.orig_messages]


    def pop(self, show_system=False):
        self.messages = self.messages[:-1]
        return self.get_html(show_system=show_system)


    def send(self, text, speak_free=True):
        if speak_free:
            text = "you can speak freely without using the format.\n" + text
        new_msg = HumanMessage(content=text)
        self.messages = self.messages + [new_msg]
        response = self.llm.invoke(self.messages)
        ai_message = AIMessage(content=response.content)
        self.messages = self.messages + [ai_message]
        return HTML(self.message_html(ai_message))
    

    def __call__(self, text, speak_free=True):
        return self.send(text, speak_free=speak_free)
        

    def message_html(self, message):
        
        message_html = message.content.replace("\n", "<br>")
        if message.type == "human":
            label = """<span style="padding: 3px; background: blue; color: white;" >human</span>"""
        elif message.type == "system":
            label = """<span style="padding: 3px; background: green; color: white;" >system</span>"""
        elif message.type == "ai":
            label = """<span style="padding: 3px; background: red; color: white;" >ai</span>"""
        else:
            label = """<span style="padding: 3px; background: gray; color: white;" >unknown</span>"""
        return f"""
<div style="border: 1px solid;">
{label}
<p style="width: 600px; font-size: 14px;">
             {message_html}
</p>
</div>
"""


    def get_html(self, show_system=True):
        output_html = ""
        for msg in self.messages:
            if show_system == False and msg.type == "system":
                continue
            output_html += self.message_html(msg)
        return output_html
        
    def _repr_html_(self):
        return self.get_html(show_system=False)


class PromptDebuger:
    def __init__(self, store=None, run_id=None, model=None):
        self.store = store
        self.model = model
        self.run_id = run_id
        self.prompt_trancer = PromptTracer()
        self.run = self.prompt_trancer.get_run(run_id)


    def get_run_conversation(self, indexs):
        curr_run = self.run
        for i in indexs:
            curr_run = curr_run[i]
        messages = curr_run.get_messages()
        return ConversationDebug(messages, model=self.model)

    def get_conversation(self, sent_idx, step_idx):
        conversation = self.store.state_list()[sent_idx].history[step_idx]['thought']
        completion = conversation.completion
        messages = conversation.messages
        messages = messages + [AIMessage(content=completion)]
        return ConversationDebug(messages, model=self.model)
    
    def show_conversations(self, sent_idx, show_system=False):
        out_html = ""
        for i, step in enumerate(self.store.state_list()[sent_idx].history):
            conversation = self.get_conversation(sent_idx, i)
            out_html += f"""
            <div> 
            <h2>Step {i}</h2>
            {conversation.get_html(show_system=show_system)}
            </div>
"""

        return HTML(f"""
<div style="display: flex">
{out_html}
</div>
""")
