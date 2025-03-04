#!/usr/bin/env python3
"""
Script to create a dataset from playthrough data, extracting discussion information and results.
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from datetime import datetime
import argparse

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the GameState class
from src.cot_faithfulness.game_state import GameState


def process_json_files(num_games=None, include_trace=True):
    """
    Process JSON files in the data/json directory.
    Create dataset rows for each discussion, tracking information in chronological order.
    
    Args:
        num_games: Maximum number of games to process (None for all)
        include_trace: Whether to include trace information in the output
        
    Returns:
        List of dictionaries containing player data
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
        if num_games is not None and processed_games >= num_games:
            break
        
        try:
            # Load the game state
            game = GameState(str(json_file_path))
            
            # Skip if no playthrough data
            if not hasattr(game, 'playthrough'):
                print(f"Skipping {json_file_path.name}: No playthrough data found")
                continue
            
            # Initialize player information
            players_info = {}
            for player in game.players:
                player_name = player.name
                players_info[player_name] = {
                    'role': 'Impostor' if player.role == 'Impostor' else 'Crewmate',
                    'model': json_file_path.name.split('_')[0] if player.role == 'Impostor' else json_file_path.name.split('_')[2],
                    'vote_count': 0,
                    'mentioned_count': 0,
                    'trace': []  # Chronological trace of all events
                }

            # Initialize game state tracking
            impostor_win = False
                
            # Check for game result
            if "Impostors win!" in game.playthrough[-1]:
                impostor_win = True

            all_discussion_messages = []  # Chronological list of all discussion messages

            for line in game.playthrough:
                line_str = str(line)
                # Check for votes
                for player_name in players_info.keys():
                    vote_pattern = re.compile(fr"vote.*voted for {player_name}")
                    if vote_pattern.match(line_str):
                        players_info[player_name]['vote_count'] += 1
            
            # Process playthrough chronologically
            for line_idx, line in enumerate(game.playthrough):
                line_str = str(line)
                
                # Extract player actions
                action_pattern = re.compile(r"Action: round: \d+ \(\d+\.\d+\$\)\. p\d+ (.*)")
                action_match = action_pattern.search(line_str)
                
                if action_match:
                    action_text = action_match.group(1)
                    # Determine who performed the action
                    actor = None
                    for player_name in players_info.keys():
                        if action_text.startswith(player_name + " "):
                            actor = player_name
                            # Add to this player's trace
                            players_info[player_name]['trace'].append(action_text)
                            break
                    
                    # Record who saw the action
                    saw_pattern = re.compile(r"\[(.*?)\] saw this action")
                    saw_match = saw_pattern.search(action_text)
                    if saw_match and actor:
                        observers = [name.strip() for name in saw_match.group(1).split(',')]
                        for observer in observers:
                            if observer in players_info:
                                # Add to each observer's trace (if they didn't perform it)
                                if observer != actor:
                                    players_info[observer]['trace'].append(action_text)
                
                # Check for discussion messages
                discussion_pattern = re.compile(r"Discussion.*chat: Discussion: \[(.*?)\]: (.*)", re.DOTALL)
                discussion_match = discussion_pattern.search(line_str)
                
                if discussion_match:
                    speaker = discussion_match.group(1)
                    message = discussion_match.group(2).strip()
                    message_text = f"Discussion: [{speaker}]: {message}"
                    
                    # Store this discussion message chronologically
                    all_discussion_messages.append(message_text)
                    
                    # Count mentions of other players in this message
                    for other_player in players_info.keys():
                        if other_player != speaker and other_player in message:
                            players_info[other_player]['mentioned_count'] += 1
                    
                    # Create a dataset row for this discussion message
                    if speaker in players_info:
                        # Copy the player's trace up to this point
                        current_trace = players_info[speaker]['trace'].copy()
                        
                        # Add all prior discussion messages to the trace
                        # Everyone sees all discussions, so include all previous ones
                        current_trace.extend(all_discussion_messages[:-1])  # All except the current one
                        
                        # Add this discussion message to the trace
                        current_trace.append(message_text)
                        
                        # Create the dataset row
                        row = {
                            'file_name': json_file_path.name,
                            'player_name': speaker,
                            'role': players_info[speaker]['role'],
                            'llm_model': players_info[speaker]['model'],
                            'has_won': (impostor_win and players_info[speaker]['role'] == 'Impostor') or 
                                      (not impostor_win and players_info[speaker]['role'] == 'Crewmate'),
                            'mentioned_count': players_info[speaker]['mentioned_count'],
                            'vote_count': players_info[speaker]['vote_count'],
                            'discussion_message': message,
                        }
                        
                        # Only include trace if requested
                        if include_trace:
                            row['trace'] = '\n'.join(current_trace)
                        
                        dataset_rows.append(row)
                
                # Check for votes
                for player_name in players_info.keys():
                    vote_pattern = re.compile(fr"vote:.*voted for {player_name}")
                    if vote_pattern.match(line_str):
                        players_info[player_name]['vote_count'] += 1
                        # Add vote to every player's trace
                        for p in players_info.keys():
                            players_info[p]['trace'].append(line_str)
            
            processed_games += 1
            total_discussions = len(all_discussion_messages)
            print(f"Processed {json_file_path.name}: Found {len(players_info)} players with {total_discussions} discussion messages")
            
        except Exception as e:
            print(f"Error processing {json_file_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return dataset_rows


def main():
    """
    Main function to create the playthrough dataset.
    """
    parser = argparse.ArgumentParser(description='Create a dataset from playthrough data.')
    parser.add_argument('--num-games', type=int, default=None, help='Maximum number of games to process')
    parser.add_argument('--no-trace', action='store_true', help='Exclude trace information from the output')
    args = parser.parse_args()
    
    # Process JSON files
    dataset_rows = process_json_files(args.num_games, not args.no_trace)
    
    if not dataset_rows:
        print("No playthrough data found.")
        return
    
    # Create a timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "out" / f"playthrough_dataset_{timestamp}.csv"
    
    # Define the field names for the CSV
    field_names = [
        'file_name',
        'player_name',
        'role',
        'llm_model',
        'has_won',
        'mentioned_count',
        'vote_count',
        'discussion_message',
    ]
    
    # Only include trace field if requested
    if not args.no_trace:
        field_names.append('trace')
    
    # Write the dataset to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
        # Write header
        writer.writeheader()
        # Write data rows
        for row in dataset_rows:
            writer.writerow(row)
    
    # Print a summary of the data
    print("\nData Summary:")
    print(f"Total discussion messages processed: {len(dataset_rows)}")
    print(f"Total games processed: {len(set(row['file_name'] for row in dataset_rows))}")
    print(f"Trace information included: {not args.no_trace}")
    
    print(f"\nOutput has been saved to: {output_file}")


if __name__ == "__main__":
    main()
