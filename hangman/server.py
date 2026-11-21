"""Optional REST API around the engine, for playing from any language or machine.

    pip install fastapi uvicorn
    uvicorn hangman.server:app --port 8000

Endpoints
    POST /games                      -> start a game          {"recorded": false}
    POST /games/{game_id}/guess      -> guess a letter        {"letter": "e"}
    GET  /status                     -> win counts and success rates
    GET  /docs                       -> interactive OpenAPI docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hangman.engine import HangmanEngine

app = FastAPI(title="Hangman engine", version="1.0")
engine = HangmanEngine()


class NewGame(BaseModel):
    recorded: bool = False


class Guess(BaseModel):
    letter: str


@app.post("/games")
def new_game(body: NewGame = NewGame()) -> dict:
    return engine.new_game(recorded=body.recorded).to_dict()


@app.post("/games/{game_id}/guess")
def guess(game_id: str, body: Guess) -> dict:
    try:
        return engine.guess(game_id, body.letter).to_dict()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.get("/status")
def status() -> dict:
    return engine.status()
