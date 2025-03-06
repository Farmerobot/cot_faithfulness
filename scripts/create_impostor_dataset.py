#!/usr/bin/env python3
"""
Script to create a dataset of impostor games, with all discussion messages consolidated.
"""

import sys
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
    """Process JSON files and create dataset rows for each game."""
    json_dir = project_root / "data" / "json"
    out_dir = project_root / "out"
    out_dir.mkdir(exist_ok=True)
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return []
    
    dataset_rows = []
    processed_games = 0
    skipped_files = {"round_limit": 0, "no_discussion_start": 0, "no_discussion_messages": 0, "other": 0}
    
    for json_file_path in json_files:
        if num_games is not None and processed_games >= num_games:
            break
        
        # Skip files with "round_limit" in the name
        if "round_limit" in json_file_path.name:
            print(f"Skipping {json_file_path.name}: Contains 'round_limit' in filename")
            skipped_files["round_limit"] += 1
            continue
        
        try:
            game = GameState(str(json_file_path))
            
            # Skip files with round_of_discussion_start = 0
            if not hasattr(game, 'round_of_discussion_start') or game.round_of_discussion_start == 0:
                print(f"Skipping {json_file_path.name}: round_of_discussion_start is 0 or not present")
                skipped_files["no_discussion_start"] += 1
                continue
            
            if not hasattr(game, 'playthrough'):
                print(f"Skipping {json_file_path.name}: No playthrough data")
                skipped_files["other"] += 1
                continue
            
            # Find impostor and initialize player info
            players_info = {}
            impostor_player = None
            
            for player in game.players:
                player_name = player.name
                is_impostor = player.role == 'Impostor'
                
                players_info[player_name] = {
                    'role': 'Impostor' if is_impostor else 'Crewmate',
                    'model': json_file_path.name.split('_')[0] if is_impostor else json_file_path.name.split('_')[2],
                    'vote_count': 0,
                    'mentioned_count': 0,
                }
                
                if is_impostor:
                    impostor_player = player_name

            if not impostor_player:
                print(f"Skipping {json_file_path.name}: No impostor found")
                skipped_files["other"] += 1
                continue
                
            # Check for game result
            impostor_win = False
            if hasattr(game, 'playthrough') and "Impostors win!" in str(game.playthrough[-1]):
                impostor_win = True

            # Process playthrough
            all_discussion_messages = []
            impostor_trace = []
            
            for line in game.playthrough:
                line_str = str(line)
                
                # Skip discussion lines and player to act next lines in the trace
                if "Discussion" in line_str or "[90m Player to act next set to:" in line_str:
                    # Only process discussion messages for the all_discussion_messages field
                    discussion_match = re.search(r"Discussion.*chat: Discussion: \[(.*?)\]: (.*)", line_str, re.DOTALL)
                    if discussion_match:
                        speaker = discussion_match.group(1)
                        message = discussion_match.group(2).strip()
                        all_discussion_messages.append(f"[{speaker}]: {message}")
                        
                        # Count mentions
                        for player_name in players_info.keys():
                            if player_name != speaker and player_name in message:
                                players_info[player_name]['mentioned_count'] += 1
                    
                    # Skip adding this line to the trace
                    continue
                
                # Clean up trace lines by removing the "Action: round: X (0.XXXXX$). pX " prefix
                trace_line = line_str
                action_match = re.match(r"Action: round: \d+ \(\d+\.\d+\$\)\. p-?\d+ (.+)", line_str)
                if action_match:
                    trace_line = action_match.group(1)
                elif line_str.startswith("dead_players:"):
                    continue  # Skip these lines
                
                if trace_line:
                    impostor_trace.append(trace_line)
                
                # Check for votes - use both patterns from the playthrough dataset
                for player_name in players_info.keys():
                    if re.match(fr"vote:.*voted for {player_name}", line_str) or re.match(fr"vote .*voted for {player_name}", line_str):
                        players_info[player_name]['vote_count'] += 1
            
            # Skip games with no discussion messages
            if not all_discussion_messages:
                print(f"Skipping {json_file_path.name}: No discussion messages found")
                skipped_files["no_discussion_messages"] += 1
                continue
                
            # Create the dataset row
            row = {
                'file_name': json_file_path.name,
                'impostor_name': impostor_player,
                'impostor_model': players_info[impostor_player]['model'],
                'crewmate_model': next((p['model'] for p in players_info.values() if p['role'] == 'Crewmate'), "Unknown"),
                'impostor_won': impostor_win,
                'impostor_mentioned_count': players_info[impostor_player]['mentioned_count'],
                'impostor_vote_count': players_info[impostor_player]['vote_count'],
                'all_discussion_messages': '\n'.join(all_discussion_messages),
            }
            
            if include_trace:
                row['impostor_trace'] = '\n'.join(impostor_trace)
            
            dataset_rows.append(row)
            processed_games += 1
            print(f"Processed {json_file_path.name}: Found {len(all_discussion_messages)} discussion messages")
            
        except Exception as e:
            print(f"Error processing {json_file_path.name}: {str(e)}")
            skipped_files["other"] += 1
            import traceback
            traceback.print_exc()
    
    # Print skipped files summary
    print("\nSkipped Files Summary:")
    print(f"  Files with 'round_limit' in name: {skipped_files['round_limit']}")
    print(f"  Files with round_of_discussion_start = 0: {skipped_files['no_discussion_start']}")
    print(f"  Files with no discussion messages: {skipped_files['no_discussion_messages']}")
    print(f"  Files skipped for other reasons: {skipped_files['other']}")
    
    return dataset_rows


def main():
    """Main function to create the impostor dataset."""
    parser = argparse.ArgumentParser(description='Create a dataset of impostor games with consolidated discussion messages.')
    parser.add_argument('--num-games', type=int, default=None, help='Maximum number of games to process')
    parser.add_argument('--no-trace', action='store_true', help='Exclude trace information from the output')
    args = parser.parse_args()
    
    dataset_rows = process_json_files(args.num_games, not args.no_trace)
    
    if not dataset_rows:
        print("No suitable game data found.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "out" / f"impostor_dataset_{timestamp}.csv"
    
    field_names = [
        'file_name', 'impostor_name', 'impostor_model', 'crewmate_model',
        'impostor_won', 'impostor_mentioned_count', 'impostor_vote_count',
        'all_discussion_messages',
    ]
    
    if not args.no_trace:
        field_names.append('impostor_trace')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow(row)
    
    # Print a sample of the first row's data if available
    if dataset_rows:
        first_row = dataset_rows[0]
        print("\nSample of first row data:")
        print(f"File: {first_row['file_name']}")
        print(f"Impostor: {first_row['impostor_name']} (Model: {first_row['impostor_model']})")
        print(f"Impostor won: {first_row['impostor_won']}")
        print(f"Impostor mentioned count: {first_row['impostor_mentioned_count']}")
        print(f"Impostor vote count: {first_row['impostor_vote_count']}")
        
        # Print a sample of the discussion messages (first 3 if available)
        messages = first_row['all_discussion_messages'].split('\n')
        print(f"Discussion messages (sample of {len(messages)} total):")
        for i, msg in enumerate(messages[:3]):
            # Truncate very long messages for display
            if len(msg) > 100:
                msg = msg[:97] + "..."
            print(f"  {i+1}. {msg}")
        if len(messages) > 3:
            print(f"  ... and {len(messages)-3} more messages")
        
        if 'impostor_trace' in first_row:
            trace_lines = first_row['impostor_trace'].split('\n')
            print(f"Trace (sample of {len(trace_lines)} total lines):")
            for i, line in enumerate(trace_lines[:3]):
                if len(line) > 100:
                    line = line[:97] + "..."
                print(f"  {i+1}. {line}")
            if len(trace_lines) > 3:
                print(f"  ... and {len(trace_lines)-3} more lines")
    
    print("\nData Summary:")
    print(f"Total games processed: {len(dataset_rows)}")
    print(f"Games where impostor won: {sum(1 for row in dataset_rows if row['impostor_won'])}")
    print(f"Trace information included: {not args.no_trace}")
    print(f"\nOutput has been saved to: {output_file}")


if __name__ == "__main__":
    main()