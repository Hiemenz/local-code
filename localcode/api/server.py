"""Optional HTTP API (install with: pip install 'localcode[api]').

Exposes the same agent over NDJSON streaming so editors and scripts can
drive localcode programmatically:

    POST /v1/message   {"content": "..."}   -> NDJSON stream of agent events
    GET  /v1/sessions                       -> session list
    GET  /v1/health
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from localcode.errors import LocalCodeError

if TYPE_CHECKING:
    from localcode.agent.loop import Agent
    from localcode.session.store import SessionStore


def create_app(agent: Agent, store: SessionStore) -> Any:
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as e:
        raise LocalCodeError(
            "API server requires extras: pip install 'localcode[api]'"
        ) from e

    app = FastAPI(title="localcode", version="0.1.0")

    class MessageIn(BaseModel):
        content: str

    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": agent.provider.model}

    @app.get("/v1/sessions")
    async def sessions() -> list[dict[str, Any]]:
        return [m.__dict__ for m in store.list_sessions()]

    @app.post("/v1/message")
    async def message(body: MessageIn) -> StreamingResponse:
        async def stream() -> Any:
            async for event in agent.run_turn(body.content):
                payload: dict[str, Any] = {"kind": event.kind}
                if event.text:
                    payload["text"] = event.text
                if event.tool_call is not None:
                    payload["tool"] = {
                        "name": event.tool_call.name,
                        "arguments": event.tool_call.arguments,
                    }
                if event.result is not None:
                    payload["result"] = {"ok": event.result.ok, "output": event.result.output}
                if event.stats:
                    payload["stats"] = event.stats
                yield json.dumps(payload, default=str) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def serve(agent: Agent, store: SessionStore, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise LocalCodeError(
            "API server requires extras: pip install 'localcode[api]'"
        ) from e
    uvicorn.run(create_app(agent, store), host=host, port=port, log_level="warning")
