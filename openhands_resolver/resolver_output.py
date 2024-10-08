from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openhands_resolver.github_issue import GithubIssue
from openhands_resolver.github_pr_review import GithubPRReview


class ResolverOutput(BaseModel):
    issue: Optional[GithubIssue] = None
    pr_review: Optional[GithubPRReview] = None
    instruction: str
    base_commit: str
    git_patch: str
    history: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    success: bool
    success_explanation: str
    error: Optional[str] = None

    @property
    def identifier(self) -> str:
        if self.issue:
            return f"issue_{self.issue.number}"
        elif self.pr_review:
            return f"pr_{self.pr_review.pr_number}_review_{self.pr_review.review_id}"
        else:
            raise ValueError("Either issue or pr_review must be set")
