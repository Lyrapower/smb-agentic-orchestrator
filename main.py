from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from audit import append_audit_entry, get_latest_audit_entries
from router import route_intent


Intent = Literal["schedule", "reschedule", "cancel", "no_action"]

app = FastAPI(title="Chat Audit API Service")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message text")


class ChatResponse(BaseModel):
    intent: Intent
    reason: str
    text: str


class ExecuteRequest(BaseModel):
    intent: Intent
    text: str = Field(..., min_length=1)


class ExecuteResponse(BaseModel):
    status: str
    intent: Intent
    result: str
    audit_entry: dict


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    intent, reason = route_intent(payload.text)
    return ChatResponse(intent=intent, reason=reason, text=payload.text)


@app.post("/execute", response_model=ExecuteResponse)
def execute(payload: ExecuteRequest) -> ExecuteResponse:
    result = {
        "schedule": "Simulated scheduling action completed.",
        "reschedule": "Simulated rescheduling action completed.",
        "cancel": "Simulated cancellation action completed.",
        "no_action": "No action performed.",
    }[payload.intent]
    saved_entry = append_audit_entry(
        {
            "intent": payload.intent,
            "text": payload.text,
            "result": result,
        }
    )
    return ExecuteResponse(
        status="ok",
        intent=payload.intent,
        result=result,
        audit_entry=saved_entry,
    )


@app.get("/audit")
def audit() -> list[dict]:
    return get_latest_audit_entries(limit=50)
