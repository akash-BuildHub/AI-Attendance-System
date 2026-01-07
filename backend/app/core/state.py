from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class CameraMeta:
    camera_id: int
    camera_name: str
    rtsp_url: str
    running: bool = False

@dataclass
class AppState:
    session_manager: Optional[Any] = None
    detector: Optional[Any] = None
    embedder: Optional[Any] = None
    config_cache: Dict[str, Any] = field(default_factory=dict)

state = AppState()
