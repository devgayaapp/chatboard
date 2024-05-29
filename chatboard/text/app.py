from typing import List, Optional
from chatboard.text.app_manager import app_manager
from chatboard.text.llms.prompt_tracer import PromptTracer
from chatboard.text.vectors.rag_documents2 import RagDocuments
from pydantic import BaseModel




class GetRagParams(BaseModel):
    namespace: str


def add_chatboard(app):
    
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
    

