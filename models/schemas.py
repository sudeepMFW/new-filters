from pydantic import BaseModel
from typing import Optional

class ImageInput(BaseModel):
    image_url: Optional[str] = None


class CompareImages(BaseModel):
    image_url_1: str
    image_url_2: str


class TextMode(BaseModel):
    mode: str = "blur"
