import asyncio
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
from mistralai import Mistral
import websockets

# ------------------------------------------------------------
# 1) Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 2) Load Environment Variables
# ------------------------------------------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing. Please set it in your .env file.")

# Choose your Mistral model
MISTRAL_MODEL = "mistral-large-latest"  # Or "mistral-small", etc.

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ------------------------------------------------------------
# 3) Game Context
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
After the move, the turn goes to the next player, or repeats the same player if conditions require another roll (like rolling a 6).
"""

# ------------------------------------------------------------
# 4) Move History
# ------------------------------------------------------------
# We'll store all moves in chronological order so the LLM has full context.
move_history = []


# ------------------------------------------------------------
# 5) Prompt-Engineering Function
# ------------------------------------------------------------
def build_prompt_for_llm(new_move: dict, next_player_id: int) -> str:
    """
    Build a well-structured prompt for the LLM, including:
    - Game context/rules
    - Complete history of moves so far
    - The most recent move
    - The next player to move
    - Request for analysis and advice (recommendation).
    """

    # Convert the entire move_history to a user-friendly JSON string
    # so the LLM can see all prior moves.
    history_json = json.dumps(move_history, indent=2)

    # Also pretty-print the latest move
    latest_move_json = json.dumps(new_move, indent=2)

    # Prompt structure
    prompt = f"""
{GAME_CONTEXT}

This is the history of moves so far in the game (chronologically):
{history_json}

The most recent move is:
{latest_move_json}

The next player to act is: Player {next_player_id}

Please provide the following:
1. A brief analysis of the game situation (from the move history).
2. A recommendation for Player {next_player_id} regarding their best move or strategy. 
   If the last move was illegal, explain why and how to fix it.

Format your response in JSON with these fields:
- "analysis": [One or two sentences summarizing what's happening]
- "recommendation": [Advice or recommended action for the next player]
- "rationale": [Brief explanation for the advice]
"""
    return prompt


# ------------------------------------------------------------
# 6) LLM Interaction
# ------------------------------------------------------------
async def get_advice_from_llm(new_move: dict, next_player_id: int) -> dict:
    """
    - Constructs a prompt including the entire move_history, the last move, and the next player.
    - Calls the Mistral LLM to get analysis and recommendation in JSON form.
    - Returns a dictionary with {analysis, recommendation, rationale}, if possible.
    """
    try:
        # 1) Build the prompt
        prompt = build_prompt_for_llm(new_move, next_player_id)

        # 2) Send to Mistral
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        advice_text = response.choices[0].message.content.strip()
        logger.info(f"Raw LLM Response:\n{advice_text}")

        # 3) Attempt to parse the LLM's JSON if it returns valid JSON
        try:
            # The model might not always return perfect JSON, so we wrap in try-except.
            advice_dict = json.loads(advice_text)
            return advice_dict
        except json.JSONDecodeError:
            # If we fail to parse, we just return the raw text
            logger.warning("LLM response is not valid JSON. Returning raw text.")
            return {
                "analysis": "",
                "recommendation": advice_text,
                "rationale": "Raw text was returned. Could not parse JSON."
            }

    except Exception as e:
        logger.error(f"Error in get_advice_from_llm: {str(e)}")
        return {
            "analysis": "",
            "recommendation": f"Error encountered: {str(e)}",
            "rationale": ""
        }


# ------------------------------------------------------------
# 7) SwarmZero Agent Setup
# ------------------------------------------------------------
sdk_context = SDKContext(config_path="./swarmzero_config.toml")


# We'll define a function for the agent that processes a move
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
    This function is triggered when a move event is received.
    - We add the move to the global move_history.
    - We call the LLM to get advice for the next player to act.
    """
    # Build a dictionary representing the move
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

    # 1) Add this move to the global move_history
    move_history.append(new_move)

    # 2) Use the LLM to provide an analysis & recommendation for next_player_id
    advice_dict = await get_advice_from_llm(new_move, next_player_id)

    # Log or return the advice
    logger.info(f"LLM Advice (JSON parsed if possible): {advice_dict}")
    return advice_dict


# Create the Agent
from swarmzero import Agent
agent = Agent(
    name="marquis_advisor",
    instruction=GAME_CONTEXT,
    sdk_context=sdk_context,
    functions=[handle_move]  # The agent can call handle_move
)


# ------------------------------------------------------------
# 8) WebSocket Listener
# ------------------------------------------------------------
async def listen_to_game_events():
    """
    Connects to ws://127.0.0.1:8080/ws and listens indefinitely.
    Each time a message is received, parse it and call handle_move().
    """
    uri = "ws://127.0.0.1:8080/ws"
    logger.info(f"Connecting to WebSocket: {uri}")

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("WebSocket connection established.")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.info(f"Received WebSocket data: {data}")

                    # 'data' typically looks like:
                    # {
                    #   "event": "play_move",
                    #   "data": "{\"session_id\":\"6\",\"player_id\":1, ...}"
                    # }
                    move_data_str = data.get("data", "{}")
                    move_data = json.loads(move_data_str) if move_data_str else {}

                    # Call the agent function to process the new move
                    # We'll try to pass in any needed fields, with defaults if missing
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
                    
                    next_player_id = (int(move_data.get("player_id", 0)) + 1)%4

                    logger.info(f"Advice for {next_player_id} Player: {advice}")
                    
                    # print(f"Final Advice for {next_player_id} Player: {advice}")
                    
                    # This is the single output for the newly received move.
                    # If needed, you could forward this advice to a UI or
                    # another WebSocket endpoint.

        except (ConnectionRefusedError, websockets.WebSocketException) as e:
            logger.warning(f"WebSocket connection failed: {str(e)}. Retrying in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"Unexpected error in listen_to_game_events: {str(e)}")
            await asyncio.sleep(3)


# ------------------------------------------------------------
# 9) Main Entry Point
# ------------------------------------------------------------
async def main():
    logger.info("Starting Ludo Advisor with full move history context...")

    # The agent.run() typically blocks, so run it in a thread executor
    loop = asyncio.get_event_loop()
    server_task = loop.run_in_executor(None, agent.run)  # HTTP server at http://localhost:8000

    # WebSocket listener runs as a parallel task
    ws_task = asyncio.create_task(listen_to_game_events())

    # Wait for both tasks
    await asyncio.gather(server_task, ws_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Ludo Advisor.")
