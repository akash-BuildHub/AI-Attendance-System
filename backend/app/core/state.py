from dataclasses import dataclass
from typing import Optional

@dataclass
class CameraMeta:
    camera_id: int
    camera_name: str
    rtsp_url: str
    running: bool = False

