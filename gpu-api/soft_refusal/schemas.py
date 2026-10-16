from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    # An earlier reply can be empty (immediate end of turn); keep it as recorded.
    content: str = Field(max_length=12000)

    @model_validator(mode="after")
    def validate_content(self):
        if self.role != "assistant" and not self.content:
            raise ValueError("User and system messages need text")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["gemma", "hermes"] = "gemma"
    messages: list[Message] = Field(min_length=1, max_length=64)
    after_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0, ge=0, le=1.5)
    top_p: float = Field(default=0.95, gt=0, le=1)
    seed: int = Field(default=0, ge=0, le=2147483647)
    stream: Literal[False] = False
    # Zero-based decoder blocks whose resid_post output is saved at the final
    # formatted-prompt token, during generation prefill. None records nothing.
    record_layers: list[int] | None = Field(default=None, min_length=1, max_length=8)
    # Live token pieces, layer pass and next-token candidates for the recording room.
    trace: bool = False

    @model_validator(mode="after")
    def validate_conversation(self):
        if self.record_layers is not None and (
                len(set(self.record_layers)) != len(self.record_layers)
                or any(not 0 <= layer < 128 for layer in self.record_layers)):
            raise ValueError("record_layers must be unique zero-based decoder block indices")
        if self.after_job_id and (len(self.messages) != 1 or self.messages[0].role != "user"):
            raise ValueError("A queued follow-up must contain one new user message")
        if sum(len(m.content) for m in self.messages) > 24000:
            raise ValueError("Combined message text exceeds 24000 characters")
        sequence = list(self.messages)
        if sequence[0].role == "system":
            sequence = sequence[1:]
        if not sequence or sequence[-1].role != "user":
            raise ValueError("Conversation must end with a user message")
        if any(m.role != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(sequence)):
            raise ValueError("Use one optional system message, then alternating user/assistant turns")
        return self


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["flux"] = "flux"
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=1024, multiple_of=64)
    height: int = Field(default=1024, ge=256, le=1024, multiple_of=64)
    # Distilled checkpoint's published baseline: four steps, guidance 1.0.
    steps: Literal[4] = 4
    seed: int = Field(default=0, ge=0, le=2147483647)
    lora: Literal["none", "body-shape-v1"] = "none"
    lora_strength: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False, strict=True)
