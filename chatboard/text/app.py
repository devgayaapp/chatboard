from typing import Any, List, Optional
from chatboard.text.app_manager import app_manager
from chatboard.text.llms.completion_parsing import is_list_model, unpack_list_model
from chatboard.text.llms.prompt_tracer import PromptTracer
from chatboard.text.vectors.rag_documents2 import RagDocuments
from pydantic import BaseModel
import asyncio



class GetRagParams(BaseModel):
    namespace: str


class UpsertRagParams(BaseModel):
    namespace: str
    input: Any
    output: Any
    id: str | int | None = None


class DeleteRagParams(BaseModel):
    id: str | int




def add_chatboard(app, rag_namespaces=None):

    if rag_namespaces:
        loop = asyncio.get_event_loop()
        for space in rag_namespaces:
            # rag_space = RagDocuments(space["namespace"], metadata_class=space["output_class"])
            # loop.run_until_complete(rag_space.verify_namespace())
            app_manager.register_rag_space(space["namespace"], space["output_class"], space["prompt"])

    # @app.on_event("startup")
    # async def init_rag():
    #     tasks = []
    #     if rag_namespaces:
    #         for space in rag_namespaces:
    #             rag_space = RagDocuments(space["namespace"], metadata_class=space["output_class"])
    #             tasks.append(rag_space.verify_namespace())
    #             app_manager.register_rag_space(space["namespace"], space["output_class"], space["prompt"])
    #     await asyncio.gather(*tasks)
    
    @app.get('/chatboard/metadata')
    def get_chatboard_metadata():
        app_metadata = app_manager.get_metadata()
        return {"metadata": app_metadata}
    
    print("Chatboard added to app.")

    @app.post("/chatboard/get_rag_document")
    async def get_rag_document(body: GetRagParams):
        print(body.namespace)
        rag_cls = app_manager.rag_spaces[body.namespace]["metadata_class"]
        ns = app_manager.rag_spaces[body.namespace]["namespace"]
        rag_space = RagDocuments(ns, metadata_class=rag_cls)
        res = await rag_space.get_many(top_k=10)
        return res


    @app.post("/chatboard/upsert_rag_document")
    async def upsert_rag_document(body: UpsertRagParams):
        rag_cls = app_manager.rag_spaces[body.namespace]["metadata_class"]
        ns = app_manager.rag_spaces[body.namespace]["namespace"]
        prompt_cls = app_manager.rag_spaces[body.namespace].get("prompt", None)
        if prompt_cls is not None:
            prompt = prompt_cls()
            user_msg_content = await prompt.render_prompt(**body.input)
            
        rag_space = RagDocuments(ns, metadata_class=rag_cls)
        doc_id = [body.id] if body.id is not None else None
        key = user_msg_content
        if is_list_model(rag_cls):
            list_model = unpack_list_model(rag_cls)
            if type(body.output) == list:
                value = [list_model(**item) for item in body.output]
            else:
                raise ValueError("Output must be a list.")
        else:
            value = rag_cls(**body.output)
        res = await rag_space.add_documents([key], [value], doc_id)
        return res
    

    @app.post("/chatboard/edit_document")
    def edit_rag_document():
        return {}
    

    @app.get("/chatboard/get_runs")
    async def get_runs(limit: int = 10, offset: int = 0, runNames: Optional[List[str]] = None):
        tracer = PromptTracer()
        runs = await tracer.aget_runs(name=runNames, limit=limit)
        return [r.run for r in runs]
    

    @app.get("/chatboard/get_run_tree")
    async def get_run_tree(run_id: str):
        tracer = PromptTracer()
        run = await tracer.aget_run(run_id)
        return run
    

