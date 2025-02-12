import asyncio
import json
import logging
import os
from datetime import datetime

import aiohttp  # For sending the LLM advice back to the WebSocket
import websockets  # For receiving moves from the primary WebSocket
import httpx       # For fetching the game state
from dotenv import load_dotenv

from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
from mistralai import Mistral

# -------------------------------------------------------------------
# 1) Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 2) Load Environment Variables
# -------------------------------------------------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing. Please set it in your .env file.")

# WebSocket endpoint to listen for new moves
NEXT_PUBLIC_WS_URL = os.getenv("NEXT_PUBLIC_WS_URL")

# API endpoint to fetch game state
NEXT_PUBLIC_API_URL = os.getenv("NEXT_PUBLIC_API_URL")

# -------------------------------------------------------------------
# 3) Configure Mistral
# -------------------------------------------------------------------
# You can set MISTRAL_MODEL or keep a default
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# If you want advanced config like temperature, top_p, etc., and if Mistral supports them, 
# you could store them in environment variables. But typically Mistral’s `chat.complete`
# is simpler. We’ll leave it as is.

# -------------------------------------------------------------------
# 4) Game Context
# -------------------------------------------------------------------
GAME_CONTEXT = """
You are an AI advisor for a Ludo game. The game rules are:
- 4 players, each with 4 tokens
- A token only leaves the starting yard if the dice roll is a 6
- Players move tokens clockwise around the board
- You can capture an opponent's token by landing on it
- The first player to get all 4 tokens home (finish area) wins

Illegal moves:
- Moving a token without a valid dice roll
- Moving backwards
- Trying to move a token out of the yard without rolling a 6

In each turn, exactly one player can make one move, using the dice roll shown.
After the move, the turn goes to the next player, or repeats the same player if conditions require another roll.
"""

# -------------------------------------------------------------------
# 5) Move History
# -------------------------------------------------------------------
# We'll store only "play_move" events here, so the LLM sees relevant context
move_history = []

# -------------------------------------------------------------------
# 6) Prompt Engineering
# -------------------------------------------------------------------
def build_prompt_for_llm(new_move: dict, next_player_id: int, game_state: dict) -> str:
    """
    Builds a prompt that includes:
    - Ludo rules & context
    - Entire move history (chronologically)
    - The new move
    - The fetched game_state
    - The next player to act
    - Request for JSON output with analysis, recommendation, rationale
    """
    history_json = json.dumps(move_history, indent=2)
    latest_move_json = json.dumps(new_move, indent=2)
    game_state_json  = json.dumps(game_state, indent=2)

    prompt = f"""
{GAME_CONTEXT}

Here is the current "Game_State" from the server:
{game_state_json}

Here is the history of moves so far (chronologically):
{history_json}

The most recent move is:
{latest_move_json}

The next player to act is: Player {next_player_id}

Please respond in valid JSON with the following fields:
- "analysis": Summarize the current situation
- "recommendation": Advice or recommended action for Player {next_player_id}
- "rationale": Short explanation for why this advice is best
"""
    return prompt.strip()

# -------------------------------------------------------------------
# 7) Mistral LLM Interaction
# -------------------------------------------------------------------
async def get_advice_from_llm(new_move: dict, next_player_id: int, game_state: dict) -> dict:
    """
    1. Build the prompt with entire move_history + game_state + new_move.
    2. Call Mistral's chat.complete(...) to get a response.
    3. Try to parse the result as JSON; return a dict with analysis, recommendation, rationale.
    """
    try:
        prompt = build_prompt_for_llm(new_move, next_player_id, game_state)

        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        advice_text = response.choices[0].message.content.strip()

        logger.info(f"Raw LLM Response:\n{advice_text}")

        # Attempt to parse JSON
        try:
            advice_dict = json.loads(advice_text)
            return advice_dict
        except json.JSONDecodeError:
            logger.warning("LLM response is not valid JSON. Returning raw text.")
            return {
                "analysis": "",
                "recommendation": advice_text,
                "rationale": "Raw text was returned; could not parse JSON."
            }

    except Exception as e:
        logger.error(f"Error in get_advice_from_llm: {e}")
        return {
            "analysis": "",
            "recommendation": f"Error encountered: {str(e)}",
            "rationale": ""
        }

# -------------------------------------------------------------------
# 8) SwarmZero Setup (Agent + handle_move)
# -------------------------------------------------------------------
sdk_context = SDKContext(config_path="./swarmzero_config.toml")

async def handle_move(
    session_id: str,
    player_id: int,
    token_id: str,
    steps: int,
    next_player_id: int,
    next_nonce: int,
    transaction_hash: str,
    session_win_result
):
    """
    Called whenever we receive an event "play_move".
    1) Fetch the game state from NEXT_PUBLIC_API_URL + game/session/{session_id}
    2) Append this move to move_history
    3) Call Mistral LLM for advice
    """
    new_move = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "player_id": player_id,
        "token_id": token_id,
        "steps": steps,
        "next_player_id": next_player_id,
        "next_nonce": next_nonce,
        "transaction_hash": transaction_hash,
        "session_win_result": session_win_result
    }

    # 1) Fetch game state
    game_state_url = f"{NEXT_PUBLIC_API_URL}game/session/{session_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(game_state_url, timeout=5)
            resp.raise_for_status()
            game_state = resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch game state from {game_state_url}: {e}")
        game_state = {"error": str(e)}

    # 2) Add move to move_history
    move_history.append(new_move)

    # 3) Query Mistral for advice
    advice_dict = await get_advice_from_llm(new_move, next_player_id, game_state)
    logger.info(f"Mistral Advice (JSON parsed if possible): {advice_dict}")
    return advice_dict

agent = Agent(
    name="marquis_advisor",
    instruction=GAME_CONTEXT,
    sdk_context=sdk_context,
    functions=[handle_move]
)

# -------------------------------------------------------------------
# 9) WebSocket Listener
# -------------------------------------------------------------------
async def listen_to_game_events():
    """
    Connect to NEXT_PUBLIC_WS_URL and listen for messages indefinitely.
    For each 'play_move' event, handle it and then send LLM advice back.
    """
    logger.info(f"Connecting to WebSocket: {NEXT_PUBLIC_WS_URL}")

    while True:
        try:
            async with websockets.connect(NEXT_PUBLIC_WS_URL) as websocket:
                logger.info("WebSocket connection established.")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.info(f"Received WebSocket data: {data}")

                    # Filter for 'play_move' events
                    event_type = data.get("event", "")
                    if event_type != "play_move":
                        logger.info(f"Ignoring event '{event_type}' since it's not 'play_move'.")
                        continue

                    # Parse the move data
                    move_data_str = data.get("data", "{}")
                    move_data = json.loads(move_data_str) if move_data_str else {}
                    
                    next_player_id = int(move_data.get("next_player_id", 0))
                    
                    if next_player_id != 0:
                        continue

                    # Call handle_move to get LLM advice
                    advice = await handle_move(
                        session_id=move_data.get("session_id", ""),
                        player_id=int(move_data.get("player_id", 0)),
                        token_id=move_data.get("token_id", ""),
                        steps=int(move_data.get("steps", 0)),
                        next_player_id=int(move_data.get("next_player_id", 0)),
                        next_nonce=int(move_data.get("next_nonce", 0)),
                        transaction_hash=move_data.get("transaction_hash", ""),
                        session_win_result=move_data.get("session_win_result", None)
                    )

                    # Example: If we want to label the event with the next player
                    logger.info(f"Advice for Player {next_player_id}: {advice}")

                    # ------------------------------------------------
                    # Send the LLM advice BACK to the server
                    # using AIOHTTP to open a new WS connection
                    # ------------------------------------------------
                    advice_message = {
                        "event": f"advisor_recommendation_for_Player_{next_player_id}",
                        "data": json.dumps(advice)
                    }
                    send_ws_url = NEXT_PUBLIC_WS_URL  # same or a different WS if needed

                    # Open a separate WS connection via aiohttp to send the advice
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(send_ws_url) as ws:
                            await ws.send_json(advice_message)

                    logger.info(f"Sent LLM advice back to the server: {advice_message}")

        except (ConnectionRefusedError, websockets.WebSocketException) as wex:
            logger.warning(f"WebSocket connection failed: {wex}. Retrying in 1s...")
            await asyncio.sleep(1)
        except Exception as ex:
            logger.error(f"Unexpected error in listen_to_game_events: {ex}")
            await asyncio.sleep(1)

# -------------------------------------------------------------------
# 10) Main Entry Point
# -------------------------------------------------------------------
async def main():
    logger.info("Starting Mistral-based Ludo Advisor with enhancements...")

    # agent.run() is blocking, so run it in a background thread
    loop = asyncio.get_event_loop()
    server_task = loop.run_in_executor(None, agent.run)

    # Meanwhile, run the WebSocket listener
    ws_task = asyncio.create_task(listen_to_game_events())

    await asyncio.gather(server_task, ws_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Ludo Advisor.")
