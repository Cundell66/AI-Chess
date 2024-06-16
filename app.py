from typing import Annotated
from autogen import ConversableAgent
from autogen import register_function
import chess
import chess.svg
from IPython.display import display
# from flask  import Flask, render_template
# from groq import Groq
import os
from dotenv import load_dotenv


# app = Flask(__name__)

# llm_config = {"model": "llama3-70b-8192"}
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# client = Groq(api_key=api_key)

# llm_config = {"config_list":[{"model": "qwen2", "base_url":"http://127.0.0.1:11434/v1", "api_key": "ollama"}]}

llm_config = {"config_list":[{"model": "gpt-4o", "api_key": api_key}]}

# Initialise the chess board
board = chess.Board()
made_move = False

def make_board(board):
    def update_board(board):
        # Create an SVG image of the board
        svg_board = chess.svg.board(board, size=400)
        with open("board.svg", "w") as svg_file:
            svg_file.write(svg_board)
    update_board(board)

# Tool for getting legal moves
def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Posible moves are: " + ",".join(
        [str(move) for move in board.legal_moves]
    )


# Tool for making a move on the board
def make_move(
        move: Annotated[str, "A move in UCI format."]
) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True

    #Get the piece name
    piece = board.piece_at(move.to_square)
    if piece is None:
        piece_name = "None"
        piece_symbol = ""
    else:
        piece_symbol = piece.symbol()
        piece_type = piece.piece_type
        if piece_type == chess.PAWN:
            piece_name = "Pawn"
        elif piece_type == chess.KNIGHT:
            piece_name = "Knight"
        elif piece_type == chess.BISHOP:
            piece_name = "Bishop"
        elif piece_type == chess.ROOK:
            piece_name = "Rook"
        elif piece_type == chess.QUEEN:
            piece_name = "Queen"
        elif piece_type == chess.KING:
            piece_name = "King"
    # embed_kernel()
    display(
        chess.svg.board(board, arrows=[(move.from_square, move.to_square)], fill={move.from_square:"gray"}, size=400)
    )
    return f"Moved {piece_name} ({piece_symbol}) from "\
    f"{chess.SQUARE_NAMES[move.from_square]} to "\
    f"{chess.SQUARE_NAMES[move.to_square]}."

# Create Agents
# Player White Agent
player_white = ConversableAgent(
    name = "Player White",
    system_message="You are a chess player and you play as white. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move."
    "If there are no legal moves left then your opponent has won",
    llm_config=llm_config["config_list"][0],
)

# Player Black Agent
player_black = ConversableAgent(
    name = "Player Black",
    system_message="You are a chess player and you play as black. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move."
    "If there are no legal moves left then your opponent has won",
    llm_config=llm_config["config_list"][0],
)

def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        make_board(board)
        return True
    else:
        return False
    
board_proxy = ConversableAgent(
    name = "Board Proxy",
    llm_config = False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

# Register the tools
for caller in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=caller,
        executor=board_proxy,
        name="get_legal_moves",
        description="Get legal moves",
    )
    register_function(
        make_move,
        caller=caller,
        executor=board_proxy,
        name="make_move",
        description="Call this tool to make a move",
    )
 
# Register the nested chats
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_white,
        "summary_method": "last_msg",
    }],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_black,
        "summary_method": "last_msg",
    }],
)

chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=100,
)


