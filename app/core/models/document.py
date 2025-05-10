from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict
    score: Optional[float] = None
    distance: Optional[float] = None
    case_summary: str = "No summary"
    legal_issues: str = "No issues"
    court_reasoning: str = "No reasoning"
    decision: str = "No decision"
    relevant_laws: str = "No laws"