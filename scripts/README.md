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
# Process up to 10 games
python scripts/create_playthrough_dataset.py --num-games 10

# Process all available games
python scripts/create_playthrough_dataset.py

# Exclude trace information (smaller CSV file)
python scripts/create_playthrough_dataset.py --no-trace
```

Command-line options:
- `--num-games N`: Process at most N games
- `--no-trace`: Exclude the trace column from the output (makes the CSV file significantly smaller)

### `analyze_playthrough_dataset.py`

Analyzes a playthrough dataset to identify correlations and determine high-quality data through visualizations. The script provides insights into player behavior, model performance, and game outcomes.

Key features:
1. **Correlation Analysis** - Identifies relationships between numerical features (message length, mentioned count, vote count, etc.)
2. **Role-Based Analysis** - Compares metrics between Impostors and Crewmates
3. **Model Performance Comparison** - Evaluates different LLM models based on win rates
4. **Discussion Content Analysis** - Analyzes common words and phrases in winning vs. losing messages
5. **Quality Criteria Suggestions** - Provides data-driven recommendations for identifying high-quality data

Usage:
```bash
# Basic usage
python scripts/analyze_playthrough_dataset.py --input /path/to/playthrough_dataset.csv

# Specify custom output directory for visualizations
python scripts/analyze_playthrough_dataset.py --input /path/to/playthrough_dataset.csv --output-dir ./analysis_results

# Get more detailed information during analysis
python scripts/analyze_playthrough_dataset.py --input /path/to/playthrough_dataset.csv --verbose
```

Command-line options:
- `--input`: Path to the input CSV file (required)
- `--output-dir`: Directory to save visualizations (default: ./analysis_output)
- `--verbose`: Print detailed information about columns and missing values

Generated visualizations include:
- Correlation heatmap of numerical features
- Message length distribution by role and winning status
- Mentioned count vs. vote count scatter plot
- Word clouds for winning and losing messages
- Win rate by model
- Feature distributions by winning status
- Player interaction network visualizing player relationships and win rates
- Message complexity analysis showing vocabulary diversity patterns
- Game progression analysis tracking vote patterns over time
- Topic clustering identifying common themes in winning vs. losing messages
- Word win rate analysis highlighting words most strongly associated with winning or losing
- Cluster words win rate visualization showing how specific word patterns within topics relate to success
- Sankey diagrams mapping the flow from roles to players to outcomes

### `assess_trace_quality.py`

Analyzes the trace information (when available) to assess playthrough quality and identify patterns in successful gameplay. This script is designed to provide specific insights from the detailed chronological record of player actions and observations.

Key features:
1. **Trace Metrics Extraction** - Measures observation count, action count, movement patterns, task completions, and more
2. **Action Diversity Analysis** - Quantifies the variety of different actions taken during gameplay
3. **Role-specific Behaviors** - Compares effective patterns for Impostors vs. Crewmates
4. **Quality Filter Recommendations** - Generates specific filter criteria to extract high-quality data
5. **Trace-focused Visualizations** - Creates visualizations that highlight the relationship between game actions and outcomes

Usage:
```bash
# Basic usage
python scripts/assess_trace_quality.py --input /path/to/playthrough_dataset.csv

# Specify custom output directory for visualizations
python scripts/assess_trace_quality.py --input /path/to/playthrough_dataset.csv --output-dir ./trace_analysis
```

Command-line options:
- `--input`: Path to the input CSV file (required, must contain the 'trace' column)
- `--output-dir`: Directory to save visualizations (default: ./trace_analysis_output)

Note: This script requires the 'trace' column to be present in the dataset. If you created a dataset with the `--no-trace` flag, this analysis cannot be performed.

### Additional Analysis Dependencies

For the advanced visualizations in the analysis scripts, install the required dependencies:

```bash
# Install analysis dependencies
pip install -r scripts/requirements-analysis.txt
```

## Output

Output files are saved to the `out/` directory with a timestamp in the filename:
- `playthrough_dataset_YYYYMMDD_HHMMSS.csv`

## Troubleshooting

### Small Datasets

The analysis scripts include automated handling for small datasets, but may display warnings or fallback to alternative visualizations when there isn't enough data for certain analysis types:

- **KDE Plot Failures**: For datasets with low variation in certain features, KDE plots may fail with a "singular matrix" error. The scripts automatically fallback to histograms in these cases.
- **Topic Clustering**: Requires at least 3 different messages to function properly.
- **Network Analysis**: May produce simplified graphs with small datasets.

### Missing Packages

- **Kaleido Package**: For exporting static images from Plotly visualizations (like Sankey diagrams), you'll need the `kaleido` package. If not installed, the script will still generate HTML visualizations and provide instructions for installing kaleido.
  ```bash
  pip install kaleido
  ```

### Best Practices

- For most insightful analysis, aim to have at least 20-30 game playthroughs in your dataset
- Ensure a mix of different roles, models, and outcomes for comparative analysis
- Use the `--verbose` flag to get detailed information about your dataset structure
