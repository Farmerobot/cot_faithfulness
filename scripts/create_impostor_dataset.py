#!/usr/bin/env python3
"""
Script to create a dataset of impostor states from game JSON files.
Only processes files where round_of_discussion_start is not 0.
Outputs the state of the first impostor to a CSV file.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import argparse

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the GameState class
from src.cot_faithfulness.game_state import GameState


def find_first_impostor(game_state):
    """
    Find the first impostor in the game state.
    
    Args:
        game_state: GameState object
        
    Returns:
        The first player who is an impostor, or None if no impostor is found
    """
    if not hasattr(game_state, 'players'):
        return None
    
    for player in game_state.players:
        if hasattr(player, 'is_impostor') and player.is_impostor:
            return player
    
    return None


def process_json_files(num_games=2):
    """
    Process JSON files in the data/json directory.
    Skip files where round_of_discussion_start is 0.
    Extract the state of the first impostor.
    
    Args:
        num_games: Maximum number of games to process
        
    Returns:
        List of tuples containing (chat_messages, impostors_won)
    """
    # Get the path to the json directory
    json_dir = project_root / "data" / "json"
    
    # Create output directory if it doesn't exist
    out_dir = project_root / "out"
    out_dir.mkdir(exist_ok=True)
    
    # Find all JSON files in the directory
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return []
    
    # List to store dataset rows
    dataset_rows = []
    processed_games = 0
    
    # Process each JSON file
    for json_file_path in json_files:
        if processed_games >= num_games:
            break
        
        try:
            # Load the game state
            game = GameState(str(json_file_path))
            
            # Skip files where round_of_discussion_start is 0
            if not hasattr(game, 'round_of_discussion_start') or game.round_of_discussion_start == 0:
                print(f"Skipping {json_file_path.name}: round_of_discussion_start is 0 or not present")
                continue
            
            # Find the first impostor
            impostor = find_first_impostor(game)
            
            if impostor is None:
                print(f"Skipping {json_file_path.name}: No impostor found")
                continue
            
            # Extract chat messages
            chat_messages = []
            if hasattr(impostor, 'state') and hasattr(impostor.state, 'chat_messages'):
                chat_messages = impostor.state.chat_messages
            
            # Extract whether the impostor won (task completed)
            impostors_won = False
            if (hasattr(impostor, 'state') and 
                hasattr(impostor.state, 'tasks') and 
                len(impostor.state.tasks) > 0 and 
                hasattr(impostor.state.tasks[0], 'completed')):
                impostors_won = impostor.state.tasks[0].completed
            
            # Add row to dataset
            dataset_rows.append((impostors_won, chat_messages))
            
            processed_games += 1
            print(f"Processed {json_file_path.name}: Found impostor {impostor.name}")
            
        except Exception as e:
            print(f"Error processing {json_file_path.name}: {str(e)}")
    
    return dataset_rows


def main():
    """
    Main function to create the impostor dataset.
    """
    parser = argparse.ArgumentParser(description='Create a dataset of impostor states from game JSON files.')
    parser.add_argument('--num-games', type=int, default=2, help='Maximum number of games to process')
    args = parser.parse_args()
    
    # Process JSON files
    dataset_rows = process_json_files(args.num_games)
    
    if not dataset_rows:
        print("No impostor data found.")
        return
    
    # Create a timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "out" / f"impostor_dataset_{timestamp}.csv"
    
    # Write the dataset to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['chat_messages', 'impostors_won'])
        # Write data rows
        for chat_messages, impostors_won in dataset_rows:
            writer.writerow([json.dumps(chat_messages), impostors_won])
    
    # Print a summary of the data
    print("\nData Summary:")
    for i, (chat_messages, impostors_won) in enumerate(dataset_rows, 1):
        print(f"Game {i}:")
        print(f"  Chat Messages: {len(chat_messages)} messages")
        print(f"  Impostor Won: {impostors_won}")
        print()
    
    print(f"\nProcessed {len(dataset_rows)} games.")
    print(f"Output has been saved to: {output_file}")


if __name__ == "__main__":
    main()
