from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ImageInput(BaseModel):
    image_url: Optional[str] = None
    userId: str

class CompareImages(BaseModel):
    image_url_1: str
    image_url_2: str
    userId: str

class TextMode(BaseModel):
    mode: str = "blur"
    userId: str

class EventLogEntry(BaseModel):
    webFeatureKey: str
    eventId: str
    processingStartTime: str
    processingEndTime: str
    report: Dict[str, Any]

class MediaInfo(BaseModel):
    inputMediaURL: str
    type: str = "IMAGE"

class ProductionResponse(BaseModel):
    id: str = Field(..., alias="_id")
    mediaId: str
    userId: str
    requestType: str
    orgId: str
    eventStartTime: str
    eventEndTime: str
    eventLog: Dict[str, EventLogEntry]
    media: MediaInfo
    timedOut: bool = False
    saved_at: str = Field(default_factory=lambda: datetime.now().isoformat())
