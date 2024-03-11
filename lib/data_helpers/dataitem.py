from typing import List, Dict, Text, Optional, Union, Any
from pydantic import BaseModel, Field


class PretokenizedData(BaseModel):
    article: List[int]
    summaries: Dict[Text, List[int]]


class Comparison(BaseModel):
    preferred: Optional[int] = None


class DataItem(BaseModel):
    sample_id: str
    article: str
    summaries: Dict[Text, Text]
    comparisons: Dict[Text, Dict[Text, Any]]
    pretokenized: Optional[PretokenizedData] = None
