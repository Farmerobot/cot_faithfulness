#!/usr/bin/env python3
"""
Script to create filtered datasets for crewmates and impostors based on specific criteria.
"""

import os
import sys
import pandas as pd
import argparse
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# Define default filtering criteria with easy modification options
DEFAULT_CRITERIA = {
    # Impostor criteria
    'impostor': {
        'max_votes': 0,          # Maximum votes received (0 or 1)
        'min_mentions': 3,      # Minimum mentions across the entire game
    },
    # Crewmate criteria
    'crewmate': {
        'max_non_impostor_votes': 0,  # Maximum number of people who didn't vote for impostor
        'min_mentions': 11,           # Minimum impostor mentions across the entire game. There are only 8 games with more than 12 mentions
    }
}


def load_playthrough_dataset(file_path):
    """
    Load the playthrough dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame containing the playthrough data
    """
    print(f"Loading playthrough dataset from: {file_path}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def preprocess_dataset(df):
    """
    Preprocess the dataset to prepare for filtering.
    
    Args:
        df: Playthrough dataset DataFrame
        
    Returns:
        Preprocessed DataFrame with additional metadata
    """
    print("Preprocessing dataset...")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Group by file_name to get game-level statistics
    game_stats = df.groupby('file_name').agg({
        'player_name': 'nunique',  # Number of players in the game
        'role': lambda x: (x == 'Impostor').sum(),  # Count impostors in each game
    }).rename(columns={'player_name': 'player_count', 'role': 'impostor_count'})
    
    # Add game-level stats back to the main dataframe
    processed_df = processed_df.merge(game_stats, on='file_name', how='left')
    
    # Get max mentioned count per game for each player
    max_mentions = df.groupby(['file_name', 'player_name'])['mentioned_count'].max().reset_index()
    max_mentions = max_mentions.rename(columns={'mentioned_count': 'max_mentioned_count'})
    
    # Add max_mentioned_count back to the main dataframe
    processed_df = processed_df.merge(max_mentions, on=['file_name', 'player_name'], how='left')
    
    # Calculate the number of votes for the impostor in each game
    impostor_votes = df[df['role'] == 'Impostor'].groupby('file_name')['vote_count'].first().reset_index()
    impostor_votes = impostor_votes.rename(columns={'vote_count': 'impostor_votes'})
    
    # Add impostor_votes back to the main dataframe
    processed_df = processed_df.merge(impostor_votes, on='file_name', how='left')
    
    # Get impostor max mentioned counts per game
    impostor_mentioned = df[df['role'] == 'Impostor'].groupby('file_name')['mentioned_count'].max().reset_index()
    impostor_mentioned = impostor_mentioned.rename(columns={'mentioned_count': 'impostor_max_mentioned_count'})
    
    # Add impostor_max_mentioned_count back to the main dataframe
    processed_df = processed_df.merge(impostor_mentioned, on='file_name', how='left')
    
    return processed_df


def filter_impostor_dataset(df, criteria=None, high_discussion_games=None):
    """
    Filter the dataset to get rows matching impostor criteria.
    For games with high discussion count, the min_mentions threshold is doubled.
    
    Args:
        df: Preprocessed playthrough dataset
        criteria: Dictionary containing filtering criteria
        high_discussion_games: Set of file_names for games with high discussion count
        
    Returns:
        Filtered DataFrame containing only qualifying impostor data
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA['impostor']
    if high_discussion_games is None:
        high_discussion_games = set()
    
    print(f"Filtering impostor dataset with base criteria: {criteria}")
    if high_discussion_games:
        print(f"(Using doubled min_mentions for {len(high_discussion_games)} games with high discussion count)")
    
    # Filter for impostors only
    impostor_df = df[df['role'] == 'Impostor']
    
    # Create mask for basic criteria
    won_mask = (impostor_df['has_won'] == True)
    vote_mask = (impostor_df['vote_count'] <= criteria['max_votes'])
    
    # Process each game with appropriate mention threshold
    mention_masks = []
    for game_name, game_group in impostor_df.groupby('file_name'):
        min_mentions = criteria['min_mentions'] * 2 if game_name in high_discussion_games else criteria['min_mentions']
        game_indices = game_group.index
        # Use impostor_max_mentioned_count when available, otherwise fall back to max_mentioned_count
        if 'impostor_max_mentioned_count' in impostor_df.columns:
            game_mask = impostor_df.index.isin(game_indices) & (impostor_df['impostor_max_mentioned_count'] > min_mentions)
        else:
            game_mask = impostor_df.index.isin(game_indices) & (impostor_df['max_mentioned_count'] > min_mentions)
        mention_masks.append(game_mask)
    
    # Combine all mention masks with OR
    if mention_masks:
        mention_mask = mention_masks[0]
        for mask in mention_masks[1:]:
            mention_mask = mention_mask | mask
    else:
        mention_mask = pd.Series(False, index=impostor_df.index)
    
    # Apply all criteria
    filtered_df = impostor_df[won_mask & vote_mask & mention_mask]
    
    print(f"Original impostor rows: {len(impostor_df)}")
    print(f"Filtered impostor rows: {len(filtered_df)}")
    
    return filtered_df


def filter_crewmate_dataset(df, criteria=None, high_discussion_games=None):
    """
    Filter the dataset to get rows matching crewmate criteria.
    For games with high discussion count, the min_mentions threshold is doubled.
    
    Args:
        df: Preprocessed playthrough dataset
        criteria: Dictionary containing filtering criteria
        high_discussion_games: Set of file_names for games with high discussion count
        
    Returns:
        Filtered DataFrame containing only qualifying crewmate data
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA['crewmate']
    if high_discussion_games is None:
        high_discussion_games = set()
    
    print(f"Filtering crewmate dataset with base criteria: {criteria}")
    if high_discussion_games:
        print(f"(Using doubled min_mentions for {len(high_discussion_games)} games with high discussion count)")
    
    # Filter for crewmates only
    crewmate_df = df[df['role'] == 'Crewmate']
    
    # Get the games where crewmates won
    winning_games = crewmate_df[crewmate_df['has_won'] == True]['file_name'].unique()
    
    # Get game-level data for voting patterns
    game_data = df.groupby('file_name').agg({
        'player_count': 'first',  # Total number of players in the game
        'impostor_votes': 'first'  # Total votes received by impostor
    })
    
    # Get games where everyone or all but one person voted for impostor
    # Calculate the expected votes (everyone or all but one voting for impostor)
    game_data['expected_min_votes'] = game_data['player_count'] - criteria['max_non_impostor_votes'] - 1
    
    # Filter games where votes for impostor meet the criteria
    qualifying_games = game_data[
        (game_data.index.isin(winning_games)) &
        (game_data['impostor_votes'] >= game_data['expected_min_votes'])
    ].index.tolist()
    
    # Filter impostor mention counts for these games with per-game thresholds
    # Use impostor_max_mentioned_count when available, otherwise fall back to max_mentioned_count
    if 'impostor_max_mentioned_count' in df.columns:
        mention_data = df.groupby('file_name')['impostor_max_mentioned_count'].first()
    else:
        mention_data = df[df['role'] == 'Impostor'].groupby('file_name')['max_mentioned_count'].max()
    
    # Process each game with appropriate mention threshold
    qualifying_games_with_mentions = []
    for game in qualifying_games:
        if game in mention_data.index:
            min_mentions = criteria['min_mentions'] * 2 if game in high_discussion_games else criteria['min_mentions']
            if mention_data[game] > min_mentions:
                qualifying_games_with_mentions.append(game)
    
    # Apply all filters to get the final crewmate dataset
    filtered_df = crewmate_df[crewmate_df['file_name'].isin(qualifying_games_with_mentions)]
    
    print(f"Original crewmate rows: {len(crewmate_df)}")
    print(f"Filtered crewmate rows: {len(filtered_df)}")
    
    return filtered_df


def count_discussion_occurrences(trace_data):
    """
    Count occurrences of 'Discussion: ' in a trace.
    
    Args:
        trace_data: String containing the trace text
        
    Returns:
        Number of occurrences of 'Discussion: '
    """
    return trace_data.count("Discussion: ")


def get_games_with_high_discussion_count(df, threshold=20):
    """
    Identify games where the maximum 'Discussion: ' count exceeds the threshold.
    
    Args:
        df: Playthrough dataset DataFrame
        threshold: Threshold count of 'Discussion: ' occurrences
        
    Returns:
        Set of file_names for games that exceed the threshold
    """
    # Process each game to find discussion counts
    game_discussion_counts = {}
    for _, row in df.iterrows():
        file_name = row['file_name']
        if 'trace' in row and isinstance(row['trace'], str):
            count = count_discussion_occurrences(row['trace'])
            
            # Initialize or update max count for this game
            if file_name not in game_discussion_counts:
                game_discussion_counts[file_name] = count
            else:
                game_discussion_counts[file_name] = max(game_discussion_counts[file_name], count)
    
    # Filter games where discussion count exceeds threshold
    high_discussion_games = {game for game, count in game_discussion_counts.items() if count > threshold}
    
    if high_discussion_games:
        print(f"Found {len(high_discussion_games)} games with discussion count exceeding {threshold}")
    
    return high_discussion_games


def create_visualizations(df, output_dir):
    """
    Create visualizations of the distribution of criteria used for filtering.
    
    Args:
        df: Preprocessed playthrough dataset
        output_dir: Directory to save visualizations
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nCreating visualizations in {viz_dir}...")
    
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Distribution of mention counts by role
    plt.subplot(2, 2, 1)
    sns.boxplot(x='role', y='max_mentioned_count', data=df)
    plt.title('Distribution of Max Mention Counts by Role')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mention_counts_by_role.png'))
    plt.clf()
    
    # 2. Distribution of vote counts by role
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='role', y='vote_count', data=df)
    plt.title('Distribution of Vote Counts by Role')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'vote_counts_by_role.png'))
    plt.clf()
    
    # 3. Distribution of discussion counts
    if 'trace' in df.columns:
        # Calculate discussion counts for each row
        df['discussion_count'] = df['trace'].apply(
            lambda x: count_discussion_occurrences(x) if isinstance(x, str) else 0
        )
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.histplot(df['discussion_count'], bins=20, kde=True)
        plt.title('Distribution of Discussion Counts')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'discussion_counts.png'))
        plt.clf()
        
        # 4. Scatter plot of mention counts vs discussion counts
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='discussion_count', y='max_mentioned_count', hue='role', data=df)
        plt.title('Mention Counts vs Discussion Counts by Role')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'mention_vs_discussion.png'))
        plt.clf()
    
    # 5. Impostor mentions vs votes received
    impostor_df = df[df['role'] == 'Impostor']
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='vote_count', y='max_mentioned_count', data=impostor_df)
    plt.title('Impostor Mentions vs Votes Received')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'impostor_mentions_vs_votes.png'))
    plt.clf()
    
    print(f"Visualizations created successfully in {viz_dir}")


def main():
    """
    Main function to create the filtered datasets.
    """
    parser = argparse.ArgumentParser(description='Create filtered crewmate and impostor datasets.')
    parser.add_argument('--input', type=str, required=True, help='Path to input playthrough CSV file')
    parser.add_argument('--output-dir', type=str, default='./out/filtered_datasets', help='Directory to save filtered datasets')
    parser.add_argument('--create-visualizations', action='store_true', help='Create visualizations of criteria distributions')
    
    # Optional arguments to override default criteria
    parser.add_argument('--impostor-max-votes', type=int, help=f'Maximum votes for impostor (default: {DEFAULT_CRITERIA["impostor"]["max_votes"]})')
    parser.add_argument('--impostor-min-mentions', type=int, help=f'Minimum mentions for impostor (default: {DEFAULT_CRITERIA["impostor"]["min_mentions"]})')
    parser.add_argument('--crewmate-max-non-impostor-votes', type=int, help=f'Maximum non-impostor votes (default: {DEFAULT_CRITERIA["crewmate"]["max_non_impostor_votes"]})')
    parser.add_argument('--crewmate-min-mentions', type=int, help=f'Minimum impostor mentions for crewmate games (default: {DEFAULT_CRITERIA["crewmate"]["min_mentions"]})')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Override default criteria with command line arguments if provided
    impostor_criteria = DEFAULT_CRITERIA['impostor'].copy()
    if args.impostor_max_votes is not None:
        impostor_criteria['max_votes'] = args.impostor_max_votes
    if args.impostor_min_mentions is not None:
        impostor_criteria['min_mentions'] = args.impostor_min_mentions
    
    crewmate_criteria = DEFAULT_CRITERIA['crewmate'].copy()
    if args.crewmate_max_non_impostor_votes is not None:
        crewmate_criteria['max_non_impostor_votes'] = args.crewmate_max_non_impostor_votes
    if args.crewmate_min_mentions is not None:
        crewmate_criteria['min_mentions'] = args.crewmate_min_mentions
    
    try:
        # Load the dataset
        df = load_playthrough_dataset(args.input)
        
        if df.empty:
            print("Error: Dataset is empty.")
            sys.exit(1)
        
        # Preprocess the dataset
        processed_df = preprocess_dataset(df)
        
        # Identify games with high discussion count
        high_discussion_games = get_games_with_high_discussion_count(df, threshold=20)
        
        # Filter impostor dataset
        impostor_df = filter_impostor_dataset(processed_df, impostor_criteria, high_discussion_games)
        
        # Filter crewmate dataset
        crewmate_df = filter_crewmate_dataset(processed_df, crewmate_criteria, high_discussion_games)
        
        # Save filtered datasets
        impostor_output_path = os.path.join(args.output_dir, 'impostor_dataset.csv')
        crewmate_output_path = os.path.join(args.output_dir, 'crewmate_dataset.csv')
        
        impostor_df.to_csv(impostor_output_path, index=False)
        crewmate_df.to_csv(crewmate_output_path, index=False)
        
        print(f"\nFiltered datasets created successfully!")
        print(f"Impostor dataset saved to: {impostor_output_path}")
        print(f"Crewmate dataset saved to: {crewmate_output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        original_games_count = df['file_name'].nunique()
        impostor_games_count = impostor_df['file_name'].nunique() if not impostor_df.empty else 0
        crewmate_games_count = crewmate_df['file_name'].nunique() if not crewmate_df.empty else 0
        
        print(f"Original dataset: {len(df)} rows from {original_games_count} games")
        print(f"Filtered impostor dataset: {len(impostor_df)} rows from {impostor_games_count} games")
        print(f"Filtered crewmate dataset: {len(crewmate_df)} rows from {crewmate_games_count} games")
        
        # Create visualizations if requested or by default
        create_visualizations(processed_df, args.output_dir)
        
    except Exception as e:
        print(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
