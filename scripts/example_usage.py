import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the GameState class
from src.cot_faithfulness.game_state import GameState


def main():
    """
    Example script that loads the first JSON file from data/json directory
    and prints the game state to console and to a file in the out directory.
    """
    # Get the path to the json directory
    json_dir = project_root / "data" / "json"
    
    # Create output directory if it doesn't exist
    out_dir = project_root / "out"
    out_dir.mkdir(exist_ok=True)
    
    # Create a timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"game_state_output_{timestamp}.txt"
    
    # Find the first JSON file in the directory
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        error_msg = f"Error: No JSON files found in {json_dir}"
        print(error_msg)
        with open(output_file, 'w') as f:
            f.write(error_msg)
        return
    
    # Use the first JSON file found
    json_file_path = json_files[0]
    
    # Load the game state from the file
    game = GameState(str(json_file_path))
    
    # Create a list to store output lines
    output_lines = []
    
    # Add game state information to output
    output_lines.append(f"Using JSON file: {json_file_path.name}")
    output_lines.append("\nLoaded Game State:")
    output_lines.append(str(game))
    
    # Add property access information to output
    output_lines.append("\nAccessing properties using dot notation:")
    
    # Access players information
    if hasattr(game, 'players') and len(game.players) > 0:
        first_player = game.players[0]
        output_lines.append(f"First player name: {first_player.name}")
        output_lines.append(f"First player role: {first_player.role}")
        output_lines.append(f"Is first player an impostor: {first_player.is_impostor}")
        
        # Access player state information
        if hasattr(first_player, 'state'):
            output_lines.append(f"First player location: {first_player.state.location}")
            output_lines.append(f"First player stage: {first_player.state.stage}")
            output_lines.append(f"First player life status: {first_player.state.life}")
            
            # Access tasks information
            if hasattr(first_player.state, 'tasks') and len(first_player.state.tasks) > 0:
                first_task = first_player.state.tasks[0]
                output_lines.append(f"First player's first task: {first_task.name}")
                output_lines.append(f"Is first task completed: {first_task.completed}")
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nOutput has been saved to: {output_file}")


if __name__ == "__main__":
    main()
