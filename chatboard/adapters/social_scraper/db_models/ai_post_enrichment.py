from typing import List, Optional
from pydantic import BaseModel


class PostAiEnrichment(BaseModel):
    main_theme: Optional[str] = None
    topics: Optional[List[str]] = None
    stance: Optional[str] = None
    language: Optional[str] = None
    ai_enrichment_counter: int = 0
