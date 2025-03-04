# Playthrough Dataset Creation Scripts

This directory contains scripts for creating datasets from game playthroughs.

## Scripts

### `create_impostor_dataset.py`

Creates a dataset of impostor states from game JSON files. Only processes files where round_of_discussion_start is not 0 and outputs the state of the first impostor to a CSV file.

Usage:
```bash
python scripts/create_impostor_dataset.py --num-games 10
```

### `create_playthrough_dataset.py`

Creates a dataset from game playthroughs, generating one row for each discussion message with the following fields:

1. **file_name** - filename of the game
2. **player_name** - name of the player who made the discussion comment
3. **role** - role of the player (impostor or crewmate)
4. **llm_model** - player adventure agent model
5. **has_won** - whether the player's role has won the game
6. **mentioned_count** - count of how many times player_name was mentioned in discussions
7. **vote_count** - count of how many players voted for player_name
8. **discussion_message** - the specific message from the player during discussion
9. **trace** - comprehensive playthrough record including all actions performed by the player, 
   all actions the player observed, and all discussion messages by the player

The script processes all discussion messages across all players in each game and creates a CSV file with the extracted data.

Usage:
```bash
python scripts/create_playthrough_dataset.py --num-games 10
```

To process all available games, omit the `--num-games` parameter:
```bash
python scripts/create_playthrough_dataset.py
```

## Output

Output files are saved to the `out/` directory with a timestamp in the filename:
- `playthrough_dataset_YYYYMMDD_HHMMSS.csv`
