"""Pydantic models for the API and the narrator's structured output."""
from typing import Literal, Union

from pydantic import BaseModel


class MessageIn(BaseModel):
    message: str


class NarratorMessage(BaseModel):
    type: Literal["message"] = "message"
    text: str


class NarratorTransition(BaseModel):
    type: Literal["transition"] = "transition"
    edge_id: str
    narration: str


class NarratorSteer(BaseModel):
    type: Literal["steer"] = "steer"
    text: str


NarratorResponse = Union[NarratorMessage, NarratorTransition, NarratorSteer]
