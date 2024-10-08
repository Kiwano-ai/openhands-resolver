from pydantic import BaseModel
from typing import List, Dict, Any

class GithubPRReview(BaseModel):
    owner: str
    repo: str
    pr_number: int
    review_id: int
    review_comments: List[Dict[str, Any]]
