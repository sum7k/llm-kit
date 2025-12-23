from pydantic import BaseModel


class Prompt(BaseModel):
    name: str
    version: str
    description: str
    inputs: dict[str, str]
    template: str

    class Config:
        extra = "forbid"
