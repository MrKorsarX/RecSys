from pydantic import BaseModel
from typing import List


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


class TestResponse(BaseModel):
    user_id: int
    data: dict


class BenchmarResponseF(BaseModel):
    iters: int
    mean_time: float
    std: float
