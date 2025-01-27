import aiohttp
import asyncio
import json
import logging
import os
from datetime import datetime
from textwrap import indent

# For optional Markdown display (like your snippet had):
from IPython.display import Markdown, display

from dotenv import load_dotenv
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext

import websockets
import httpx  # We'll use httpx for async requests to fetch game state

# ------------------------------------------------------------
# 1) Google's Generative AI (Gemini) imports
# ------------------------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Could not import google.generativeai. "
        "Install via: pip install google-generativeai\n"
        "Note: the code below assumes Gemini usage in a style similar to your snippet."
    )

# ------------------------------------------------------------
# 2) Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 3) Load Environment Variables
# ------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")

# Example model name from your snippet:
GEMINI_MODEL_NAME = "models/gemini-1.5-flash"

# ------------------------------------------------------------
# 4) Configure Gemini/Generative AI
# ------------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

temperature = os.getenv("TEMPERATURE")
top_p = os.getenv("TOP_P")
top_k = os.getenv("TOP_K")
max_output_tokens = os.getenv("MAX_OUTPUT_TOKENS")

# Example generation config from your snippet
generation_config = {
    "temperature": float(temperature),
    "top_p": float(top_p),
    "top_k": int(top_k),
    "max_output_tokens": int(max_output_tokens),
}

model = genai.GenerativeModel(
    GEMINI_MODEL_NAME,
    generation_config=generation_config
)

# Optional helper for turning bullets into Markdown
def to_markdown(text: str) -> Markdown:
    replaced = text.replace('•', '  *')
    return Markdown(indent(replaced, '> '))

# ------------------------------------------------------------
# 5) Ludo Game Context
# ------------------------------------------------------------
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
After the move, the turn goes to the next player (or repeats the same player if conditions require another roll).
"""

# ------------------------------------------------------------
# 6) Move History
# ------------------------------------------------------------
move_history = []

# ------------------------------------------------------------
# 7) Build Prompt
# ------------------------------------------------------------
def build_prompt_for_llm(
    new_move: dict,
    next_player_id: int,
    game_state: dict
) -> str:
    """
    Construct a thorough prompt that includes:
    - Ludo rules & context
    - The entire move history
    - The new move
    - The latest game state from the URL
    - The next player to act
    - Ask for a JSON with "analysis", "recommendation", "rationale"
    """
    history_json = json.dumps(move_history, indent=2)
    latest_move_json = json.dumps(new_move, indent=2)
    game_state_json  = json.dumps(game_state, indent=2)

    prompt = f"""
{GAME_CONTEXT}

Here is the current "Game_State" JSON from the server:
{game_state_json}

Here is the history of moves so far (chronologically):
{history_json}

The most recent move is:
{latest_move_json}

The next player to act is: Player {next_player_id}

Please respond in valid JSON with the following fields:
- "analysis": A brief summary of the current situation
- "recommendation": Advice or recommended action for Player {next_player_id}
- "rationale": A short explanation of why this is the best advice
"""
    return prompt.strip()

# ------------------------------------------------------------
# 8) Ask GEMINI for Advice
# ------------------------------------------------------------
async def get_advice_from_gemini(
    new_move: dict,
    next_player_id: int,
    game_state: dict
) -> dict:
    """
    1. Build a prompt with the entire move_history + game_state + new move.
    2. Call the Gemini model to generate content.
    3. Try to parse the result as JSON; otherwise return raw text.
    """
    try:
        prompt = build_prompt_for_llm(new_move, next_player_id, game_state)

        # Generate content with the model
        response = model.generate_content(prompt)
        advice_text = response.text  # The generated text

        logger.info(f"Raw LLM Response:\n{advice_text}")

        # Try to parse the text as JSON
        try:
            advice_dict = json.loads(advice_text)
            return advice_dict
        except json.JSONDecodeError:
            logger.warning("LLM response is not valid JSON. Returning raw text.")
            return {
                "analysis": "",
                "recommendation": advice_text,
                "rationale": "Raw text was returned. Could not parse JSON."
            }

    except Exception as e:
        logger.error(f"Error in get_advice_from_gemini: {str(e)}")
        return {
            "analysis": "",
            "recommendation": f"Error encountered: {str(e)}",
            "rationale": ""
        }

# ------------------------------------------------------------
# 9) SwarmZero Setup
# ------------------------------------------------------------
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
    Called each time a new move event is received:
      1. Fetch the "Game_State" from the URL with session_id
      2. Append this move to global move_history
      3. Query Gemini for advice
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

    # --- 1) Fetch the game state JSON from the remote URL ---
    # For example: http://127.0.0.1:8080/game/session/{session_id}
    
    public_api_url=os.getenv("NEXT_PUBLIC_API_URL")
    
    game_state_url = f"{public_api_url}game/session/{session_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(game_state_url, timeout=5)
            resp.raise_for_status()
            game_state = resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch game state from {game_state_url}: {e}")
        game_state = {"error": str(e)}

    # --- 2) Add this new move to the global move_history ---
    move_history.append(new_move)

    # --- 3) Use Gemini to get an analysis & recommendation ---
    advice_dict = await get_advice_from_gemini(new_move, next_player_id, game_state)
    logger.info(f"Gemini Advice (JSON parsed if possible): {advice_dict}")

    return advice_dict

agent = Agent(
    name="marquis_advisor",
    instruction=GAME_CONTEXT,
    sdk_context=sdk_context,
    functions=[handle_move]
)

# ------------------------------------------------------------
# 10) WebSocket Listener
# ------------------------------------------------------------
async def listen_to_game_events():
    """
    Connects to websockets and listens indefinitely.
    On each move event, parse it and call handle_move().
    """
    uri = os.getenv("NEXT_PUBLIC_WS_URL")  # to run locally use ws://127.0.0.1:8080/ws
    logger.info(f"Connecting to WebSocket: {uri}")

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("WebSocket connection established.")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.info(f"Received WebSocket data: {data}")

                    # 2) Filter out events that are not "play_move"
                    event_type = data.get("event", "")
                    if event_type != "play_move":
                        logger.info(f"Ignoring event '{event_type}' since it's not 'play_move'.")
                        continue

                    # 3) Now it's a play_move event; parse data
                    move_data_str = data.get("data", "{}")
                    move_data = json.loads(move_data_str) if move_data_str else {}

                    # Process the new move
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

                    next_player_id = int(move_data.get("next_player_id", 0))
                    
                    logger.info(f"Advice for {next_player_id} Player: {advice}")
                    
                    # print(f"Advice for {next_player_id} Player: {advice}")
                    
                    # Send the LLM advice back to the server on the same WebSocket
                    advice_message = {
                        "event": f"advisor_recommendation_for_Player_{next_player_id}",
                        "data": json.dumps(advice)  # Convert dict to JSON
                    }
                    
                    # async with websockets.connect(uri) as websocket:
                    #     await websocket.send(json.dumps(advice_message))
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(uri) as ws:
                            # Send JSON data
                            advice_message = {
                                "event": f"advisor_recommendation_for_Player_{next_player_id}",
                                "data": json.dumps(advice)  # Convert dict to JSON
                            }
                            await ws.send_json(advice_message)
                            print("Data sent using aiohttp!")
                        
                    logger.info(f"Sent LLM advice back to the server: {advice_message}")               

        except (ConnectionRefusedError, websockets.WebSocketException) as wex:
            logger.warning(f"WebSocket connection failed: {wex}. Retrying in 1s...")
            await asyncio.sleep(1)
        except Exception as ex:
            logger.error(f"Unexpected error in listen_to_game_events: {ex}")
            await asyncio.sleep(1)

# ------------------------------------------------------------
# 11) Main Entry Point
# ------------------------------------------------------------
async def main():
    logger.info("Starting Ludo Advisor with Gemini + Game_State integration...")

    # agent.run() is blocking, run it in a background thread
    loop = asyncio.get_event_loop()
    server_task = loop.run_in_executor(None, agent.run)

    # WebSocket listener runs in parallel
    ws_task = asyncio.create_task(listen_to_game_events())

    await asyncio.gather(server_task, ws_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Ludo Advisor.")
