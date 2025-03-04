#!/usr/bin/env python3
"""
Script to analyze playthrough dataset, identify correlations, and generate visualizations
to help determine high-quality data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import re
from collections import Counter
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def load_dataset(file_path):
    """
    Load and preprocess the playthrough dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed pandas DataFrame
    """
    print(f"Loading dataset from: {file_path}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create additional features
    if 'discussion_message' in df.columns:
        # Message length - handle NaN values by converting to empty string first
        df['message_length'] = df['discussion_message'].fillna('').apply(len)
        
        # Word count - handle NaN values
        df['word_count'] = df['discussion_message'].fillna('').apply(lambda x: len(str(x).split()))
        
        # Question count (sentences ending with ?) - handle NaN values
        df['question_count'] = df['discussion_message'].fillna('').apply(
            lambda x: len(re.findall(r'\?', str(x)))
        )
    
    # Convert boolean columns to int for correlation analysis
    if 'has_won' in df.columns:
        df['has_won_int'] = df['has_won'].astype(str).map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
        # Handle any missing values
        df['has_won_int'] = df['has_won_int'].fillna(0).astype(int)
    
    return df


def analyze_correlations(df):
    """
    Analyze correlations between numerical features.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Correlation matrix
    """
    print("\nAnalyzing correlations...")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numerical columns for correlation: {numerical_cols.tolist()}")
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    return corr_matrix


def role_based_analysis(df):
    """
    Analyze patterns based on player roles.
    
    Args:
        df: pandas DataFrame
    """
    print("\nRole-based analysis...")
    
    if 'role' not in df.columns:
        print("Role column not found in the dataset.")
        return
    
    # Count by role
    role_counts = df['role'].value_counts()
    print(f"Messages by role:\n{role_counts}")
    
    if 'has_won' in df.columns:
        # Win rates by role
        win_rates = df.groupby('role')['has_won_int'].mean()
        print(f"\nWin rates by role:\n{win_rates}")
    
    # Average metrics by role
    metrics = ['mentioned_count', 'vote_count', 'message_length', 'word_count', 'question_count']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        metrics_by_role = df.groupby('role')[available_metrics].mean()
        print(f"\nAverage metrics by role:\n{metrics_by_role}")


def model_based_analysis(df):
    """
    Analyze patterns based on LLM models.
    
    Args:
        df: pandas DataFrame
    """
    print("\nModel-based analysis...")
    
    if 'llm_model' not in df.columns:
        print("Model column not found in the dataset.")
        return
    
    # Count by model
    model_counts = df['llm_model'].value_counts()
    print(f"Messages by model:\n{model_counts}")
    
    if 'has_won' in df.columns:
        # Win rates by model
        win_rates = df.groupby('llm_model')['has_won_int'].mean()
        print(f"\nWin rates by model:\n{win_rates}")
    
    # Average metrics by model
    metrics = ['mentioned_count', 'vote_count', 'message_length', 'word_count', 'question_count']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        metrics_by_model = df.groupby('llm_model')[available_metrics].mean()
        print(f"\nAverage metrics by model:\n{metrics_by_model}")


def analyze_discussions(df):
    """
    Analyze discussion content to identify patterns.
    
    Args:
        df: pandas DataFrame
    """
    print("\nAnalyzing discussion content...")
    
    if 'discussion_message' not in df.columns:
        print("Discussion message column not found in the dataset.")
        return
    
    # Split into winning and losing messages
    if 'has_won' in df.columns:
        winning_messages = df[df['has_won_int'] == 1]['discussion_message']
        losing_messages = df[df['has_won_int'] == 0]['discussion_message']
        
        print(f"Number of winning messages: {len(winning_messages)}")
        print(f"Number of losing messages: {len(losing_messages)}")
        
        # Get most common words in winning vs losing messages
        def get_common_words(text_series, top_n=20):
            all_text = ' '.join(text_series.astype(str))
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            return Counter(words).most_common(top_n)
        
        print("\nMost common words in winning messages:")
        for word, count in get_common_words(winning_messages):
            print(f"  {word}: {count}")
        
        print("\nMost common words in losing messages:")
        for word, count in get_common_words(losing_messages):
            print(f"  {word}: {count}")


def generate_visualizations(df, output_dir):
    """
    Generate visualizations to help identify high-quality data.
    
    Args:
        df: pandas DataFrame
        output_dir: Directory to save visualizations
    """
    print("\nGenerating visualizations...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    # Common color palettes
    win_loss_palette = {True: "#2ecc71", False: "#e74c3c"}  # Green for win, Red for loss
    
    # Check if we have enough data for meaningful visualizations
    if len(df) < 5:
        print("Warning: Dataset is too small for some visualizations. Generating limited visualizations.")
    
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if not numerical_cols.empty:
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png')
        plt.close()
    
    # 2. Message length distribution by role and win status
    if all(col in df.columns for col in ['role', 'message_length', 'has_won']):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='role', y='message_length', hue='has_won', data=df)
        plt.title('Message Length by Role and Win Status')
        plt.tight_layout()
        plt.savefig(output_path / 'message_length_by_role_win.png')
        plt.close()
    
    # 3. Mentioned count vs vote count scatter plot
    if all(col in df.columns for col in ['mentioned_count', 'vote_count', 'has_won']):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='mentioned_count', 
            y='vote_count', 
            hue='has_won',
            size='message_length' if 'message_length' in df.columns else None,
            data=df
        )
        plt.title('Mentioned Count vs Vote Count')
        plt.tight_layout()
        plt.savefig(output_path / 'mentioned_vs_vote_count.png')
        plt.close()
    
    # 4. Word clouds for winning and losing messages
    if all(col in df.columns for col in ['discussion_message', 'has_won_int']):
        # Winning messages word cloud
        winning_text = ' '.join(df[df['has_won_int'] == 1]['discussion_message'].astype(str).fillna(''))
        if winning_text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                contour_width=3
            ).generate(winning_text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Winning Messages')
            plt.tight_layout()
            plt.savefig(output_path / 'wordcloud_winning.png')
            plt.close()
        
        # Losing messages word cloud
        losing_text = ' '.join(df[df['has_won_int'] == 0]['discussion_message'].astype(str).fillna(''))
        if losing_text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                contour_width=3
            ).generate(losing_text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Losing Messages')
            plt.tight_layout()
            plt.savefig(output_path / 'wordcloud_losing.png')
            plt.close()
    
    # 5. Win rate by model
    if all(col in df.columns for col in ['llm_model', 'has_won_int']):
        win_rates = df.groupby('llm_model')['has_won_int'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='llm_model', y='has_won_int', data=win_rates)
        plt.title('Win Rate by Model')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'win_rate_by_model.png')
        plt.close()
    
    # 6. Feature distributions by winning status
    for feature in ['mentioned_count', 'vote_count', 'message_length', 'word_count', 'question_count']:
        if all(col in df.columns for col in [feature, 'has_won']):
            # Check if we have enough variation in the data
            unique_values = df[feature].nunique()
            
            plt.figure(figsize=(10, 6))
            
            if unique_values > 1:
                try:
                    # Try with KDE first
                    sns.histplot(data=df, x=feature, hue='has_won', kde=True, element='step')
                except Exception as e:
                    print(f"Warning: Could not generate KDE plot for {feature}: {e}")
                    # Fall back to simple histogram without KDE
                    sns.histplot(data=df, x=feature, hue='has_won', kde=False, element='step')
            else:
                # Not enough variation, use barplot
                sns.countplot(data=df, x=feature, hue='has_won')
                
            plt.title(f'{feature} Distribution by Win Status')
            plt.tight_layout()
            plt.savefig(output_path / f'{feature}_by_win_status.png')
            plt.close()
    
    # 7. Network Analysis: Player Vote Relationships
    if all(col in df.columns for col in ['player_name', 'vote_count']):
        try:
            # Create a player-centric graph showing voting relationships
            plt.figure(figsize=(14, 10))
            
            # Group by player and get average vote counts and win rates
            player_stats = df.groupby('player_name').agg({
                'vote_count': 'mean',
                'has_won_int': 'mean'
            }).reset_index()
            
            # Create a graph
            G = nx.Graph()
            
            # Add nodes (players)
            for _, row in player_stats.iterrows():
                G.add_node(
                    row['player_name'], 
                    win_rate=row['has_won_int'],
                    vote_count=row['vote_count']
                )
            
            # Add edges (interactions between players)
            players = list(G.nodes())
            for i, player1 in enumerate(players):
                for player2 in players[i+1:]:
                    # Filter discussions where both players were mentioned
                    interactions = df[
                        (df['discussion_message'].str.contains(player1, na=False)) & 
                        (df['discussion_message'].str.contains(player2, na=False))
                    ]
                    
                    if len(interactions) > 0:
                        G.add_edge(player1, player2, weight=len(interactions))
            
            # Draw the graph
            pos = nx.spring_layout(G, seed=42)
            
            # Node attributes
            node_sizes = [300 * (1 + G.nodes[node]['win_rate']) for node in G.nodes()]
            node_colors = [G.nodes[node]['win_rate'] for node in G.nodes()]
            
            # Edge attributes
            edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
            
            # Create a custom colormap for win rates (red to green)
            win_rate_cmap = LinearSegmentedColormap.from_list("win_rate", ["#e74c3c", "#f39c12", "#2ecc71"])
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=win_rate_cmap, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            
            plt.title('Player Interaction Network (node size = win rate, edge thickness = interaction frequency)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path / 'player_interaction_network.png')
            plt.close()
        except Exception as e:
            print(f"Could not generate player network visualization: {e}")
    
    # 8. Message Complexity Analysis
    if all(col in df.columns for col in ['discussion_message', 'has_won']):
        try:
            # Calculate more advanced text metrics
            df['avg_word_length'] = df['discussion_message'].fillna('').apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
            )
            
            df['unique_word_ratio'] = df['discussion_message'].fillna('').apply(
                lambda x: len(set(str(x).lower().split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
            )
            
            # Create a scatter plot of message complexity metrics
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(
                data=df,
                x='avg_word_length',
                y='unique_word_ratio',
                hue='has_won',
                size='word_count',
                sizes=(20, 200),
                alpha=0.6,
                palette=win_loss_palette
            )
            
            plt.title('Message Complexity Analysis')
            plt.xlabel('Average Word Length')
            plt.ylabel('Unique Word Ratio (Vocabulary Diversity)')
            plt.tight_layout()
            plt.savefig(output_path / 'message_complexity.png')
            plt.close()
            
            # Density plot of these metrics by win status
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            sns.kdeplot(data=df, x='avg_word_length', hue='has_won', fill=True, palette=win_loss_palette)
            plt.title('Distribution of Average Word Length by Win Status')
            
            plt.subplot(1, 2, 2)
            sns.kdeplot(data=df, x='unique_word_ratio', hue='has_won', fill=True, palette=win_loss_palette)
            plt.title('Distribution of Vocabulary Diversity by Win Status')
            
            plt.tight_layout()
            plt.savefig(output_path / 'message_complexity_distribution.png')
            plt.close()
        except Exception as e:
            print(f"Could not generate message complexity visualization: {e}")
    
    # 9. Time Series Analysis: Game Progression
    if all(col in df.columns for col in ['file_name', 'vote_count']):
        try:
            # Group each game's discussion messages in sequence
            games = df['file_name'].unique()
            
            plt.figure(figsize=(14, 8))
            
            for idx, game in enumerate(games[:5]):  # Limit to first 5 games to avoid clutter
                game_data = df[df['file_name'] == game].copy()
                
                # Calculate game-specific metrics
                game_data['message_sequence'] = range(len(game_data))
                game_data['cum_vote_count'] = game_data['vote_count'].cumsum()
                
                # Plot vote accumulation over game progression
                plt.plot(
                    game_data['message_sequence'], 
                    game_data['cum_vote_count'], 
                    label=f'Game {idx+1}',
                    marker='o',
                    markersize=4,
                    alpha=0.7
                )
            
            plt.title('Vote Accumulation Throughout Game Progression')
            plt.xlabel('Message Sequence')
            plt.ylabel('Cumulative Vote Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'game_progression_votes.png')
            plt.close()
        except Exception as e:
            print(f"Could not generate game progression visualization: {e}")
    
    # 10. Text Topic Modeling with clustering
    if 'discussion_message' in df.columns and len(df) >= 10:  # Need enough data for meaningful clustering
        try:
            # Create TF-IDF features from discussion messages
            tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=3
            )
            
            # Transform discussion messages to TF-IDF features
            tfidf_features = tfidf_vectorizer.fit_transform(df['discussion_message'].fillna(''))
            
            # Reduce dimensionality for visualization
            svd = TruncatedSVD(n_components=2, random_state=42)
            reduced_features = svd.fit_transform(tfidf_features)
            
            # Cluster the documents
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(tfidf_features)
            
            # Create a DataFrame for visualization
            cluster_df = pd.DataFrame({
                'x': reduced_features[:, 0],
                'y': reduced_features[:, 1],
                'cluster': clusters,
                'has_won': df['has_won'].astype(str)
            })
            
            # Plot clusters
            plt.figure(figsize=(12, 10))
            
            # Plot by cluster
            sns.scatterplot(
                data=cluster_df,
                x='x', y='y',
                hue='cluster',
                style='has_won',
                palette='viridis',
                s=100,
                alpha=0.7
            )
            
            plt.title('Topic Clusters in Discussion Messages')
            plt.xlabel('SVD Component 1')
            plt.ylabel('SVD Component 2')
            plt.tight_layout()
            plt.savefig(output_path / 'topic_clusters.png')
            plt.close()
            
            # Get top words for each cluster
            ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = tfidf_vectorizer.get_feature_names_out()
            
            # Create a visualization of top words per cluster
            plt.figure(figsize=(15, 12))
            
            for i in range(kmeans.n_clusters):
                plt.subplot(kmeans.n_clusters, 1, i+1)
                
                # Get top 10 words for this cluster
                top_words = [terms[ind] for ind in ordered_centroids[i, :10]]
                top_weights = [kmeans.cluster_centers_[i, ind] for ind in ordered_centroids[i, :10]]
                
                # Create horizontal bar chart
                bars = plt.barh(top_words, top_weights, height=0.7)
                plt.title(f'Cluster {i+1} Top Words')
                plt.xlabel('TF-IDF Weight')
                plt.tight_layout()
                
                # Add count of winning vs losing messages in this cluster
                win_count = sum((clusters == i) & (df['has_won_int'] == 1))
                loss_count = sum((clusters == i) & (df['has_won_int'] == 0))
                win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
                
                plt.annotate(
                    f'Win rate: {win_rate:.2f} ({win_count}/{win_count+loss_count})', 
                    xy=(0.05, 0.05), 
                    xycoords='axes fraction'
                )
            
            plt.tight_layout(pad=3.0)
            plt.savefig(output_path / 'cluster_top_words.png')
            plt.close()
            
            # NEW VISUALIZATION: Word-level win rate analysis
            try:
                # Extract top 30 words from each winning and losing group
                vectorizer = CountVectorizer(stop_words='english', max_features=100)
                word_counts = vectorizer.fit_transform(df['discussion_message'].fillna(''))
                words = vectorizer.get_feature_names_out()
                
                # Create a DataFrame with word frequencies
                word_freq_df = pd.DataFrame(word_counts.toarray(), columns=words)
                word_freq_df['has_won'] = df['has_won'].values
                
                # Calculate win rate for each word
                word_win_rates = {}
                word_usage_counts = {}
                min_word_occurence = 3  # Only consider words that appear at least this many times
                
                for word in words:
                    # Get messages containing this word
                    messages_with_word = word_freq_df[word_freq_df[word] > 0]
                    total_count = len(messages_with_word)
                    
                    if total_count >= min_word_occurence:
                        win_count = messages_with_word['has_won'].sum()
                        win_rate = win_count / total_count
                        word_win_rates[word] = win_rate
                        word_usage_counts[word] = total_count
                
                # Sort words by win rate
                sorted_words = sorted(word_win_rates.items(), key=lambda x: x[1], reverse=True)
                
                # Plot top and bottom words by win rate
                plt.figure(figsize=(18, 10))
                
                # Plot top 15 words with highest win rates
                top_words = sorted_words[:15]
                bottom_words = sorted_words[-15:]
                
                # Top win rate words
                plt.subplot(1, 2, 1)
                words_list = [word for word, _ in top_words]
                win_rates = [rate for _, rate in top_words]
                
                # Create bars with color based on usage frequency
                bars = plt.barh(words_list, win_rates)
                
                # Color bars by frequency
                for i, (word, _) in enumerate(top_words):
                    count = word_usage_counts[word]
                    # Normalize count to a value between 0.3 and 1.0 for alpha
                    alpha = min(1.0, 0.3 + 0.7 * (count / max(word_usage_counts.values())))
                    bars[i].set_alpha(alpha)
                    # Annotate with count
                    plt.text(win_rates[i] + 0.01, i, f"n={count}", va='center', fontsize=8)
                
                plt.xlim(0, 1.1)
                plt.title('Words with Highest Win Rates')
                plt.xlabel('Win Rate')
                
                # Add a "success words" annotation
                plt.text(0.5, -1.5, "Words associated with winning messages", 
                         ha='center', fontsize=12, fontweight='bold')
                
                # Bottom win rate words
                plt.subplot(1, 2, 2)
                words_list = [word for word, _ in bottom_words]
                win_rates = [rate for _, rate in bottom_words]
                
                # Create bars with color based on usage frequency
                bars = plt.barh(words_list, win_rates)
                
                # Color bars by frequency
                for i, (word, _) in enumerate(bottom_words):
                    count = word_usage_counts[word]
                    # Normalize count to a value between 0.3 and 1.0 for alpha
                    alpha = min(1.0, 0.3 + 0.7 * (count / max(word_usage_counts.values())))
                    bars[i].set_alpha(alpha)
                    # Annotate with count
                    plt.text(win_rates[i] + 0.01, i, f"n={count}", va='center', fontsize=8)
                
                plt.xlim(0, 1.1)
                plt.title('Words with Lowest Win Rates')
                plt.xlabel('Win Rate')
                
                # Add a "risk words" annotation
                plt.text(0.5, -1.5, "Words associated with losing messages", 
                         ha='center', fontsize=12, fontweight='bold')
                
                plt.tight_layout(pad=3.0)
                plt.savefig(output_path / 'word_win_rates.png')
                plt.close()
                
                # Create a combined visualization of cluster words and their win rates
                if len(kmeans.cluster_centers_) > 0:
                    plt.figure(figsize=(15, 4 * kmeans.n_clusters))
                    
                    for i in range(kmeans.n_clusters):
                        plt.subplot(kmeans.n_clusters, 1, i+1)
                        
                        # Get top words for this cluster
                        top_words = [terms[ind] for ind in ordered_centroids[i, :15]]
                        
                        # Get win rates for these words
                        word_rates = []
                        word_counts = []
                        for word in top_words:
                            if word in word_win_rates:
                                word_rates.append(word_win_rates[word])
                                word_counts.append(word_usage_counts[word])
                            else:
                                word_rates.append(0)
                                word_counts.append(0)
                        
                        # Create horizontal bar chart colored by win rate
                        bars = plt.barh(top_words, word_rates, height=0.7)
                        
                        # Color bars by win rate
                        for j, bar in enumerate(bars):
                            # Set color based on win rate (green for high, red for low)
                            if word_rates[j] > 0.8:
                                bar.set_color('#2ecc71')  # Strong green
                            elif word_rates[j] > 0.6:
                                bar.set_color('#7ed6a5')  # Light green
                            elif word_rates[j] > 0.4:
                                bar.set_color('#cccccc')  # Gray
                            elif word_rates[j] > 0.2:
                                bar.set_color('#e6a8a8')  # Light red
                            else:
                                bar.set_color('#e74c3c')  # Strong red
                            
                            # Annotate with count
                            plt.text(word_rates[j] + 0.01, j, f"n={word_counts[j]}", va='center', fontsize=8)
                        
                        plt.xlim(0, 1.1)
                        plt.title(f'Cluster {i+1} Words by Win Rate')
                        plt.xlabel('Win Rate')
                        
                        # Add win rate for cluster
                        win_count = sum((clusters == i) & (df['has_won_int'] == 1))
                        loss_count = sum((clusters == i) & (df['has_won_int'] == 0))
                        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
                        
                        plt.annotate(
                            f'Cluster Win Rate: {win_rate:.2f} ({win_count}/{win_count+loss_count})', 
                            xy=(0.05, 0.05), 
                            xycoords='axes fraction',
                            fontsize=12,
                            fontweight='bold'
                        )
                    
                    plt.tight_layout(pad=3.0)
                    plt.savefig(output_path / 'cluster_words_win_rates.png')
                    plt.close()
            except Exception as e:
                print(f"Could not generate word win rate visualization: {e}")
        except Exception as e:
            print(f"Could not generate topic modeling visualization: {e}")
    
    # 11. Sankey Diagram: Player Decision Flow
    if all(col in df.columns for col in ['player_name', 'role', 'has_won']):
        fig = None  # Define fig at this scope so it's available to all blocks
        try:
            try:
                import plotly.graph_objects as go
                import kaleido  # Check if kaleido is available
                
                # Create nodes and links for Sankey diagram
                # Roles -> Players -> Outcomes
                
                # Get unique players, roles and outcomes
                players = df['player_name'].unique()
                roles = df['role'].unique()
                outcomes = ['Won', 'Lost']
                
                # Create mapping dictionaries for node indices
                role_idx = {role: i for i, role in enumerate(roles)}
                player_idx = {player: i + len(roles) for i, player in enumerate(players)}
                outcome_idx = {outcome: i + len(roles) + len(players) for i, outcome in enumerate(outcomes)}
                
                # Prepare node labels
                node_labels = list(roles) + list(players) + outcomes
                
                # Prepare links: Role -> Player
                sources = []
                targets = []
                values = []
                
                # Role -> Player links
                for player in players:
                    player_data = df[df['player_name'] == player]
                    for role in roles:
                        count = len(player_data[player_data['role'] == role])
                        if count > 0:
                            sources.append(role_idx[role])
                            targets.append(player_idx[player])
                            values.append(count)
                
                # Player -> Outcome links
                for player in players:
                    player_data = df[df['player_name'] == player]
                    win_count = player_data['has_won_int'].sum()
                    loss_count = len(player_data) - win_count
                    
                    if win_count > 0:
                        sources.append(player_idx[player])
                        targets.append(outcome_idx['Won'])
                        values.append(win_count)
                    
                    if loss_count > 0:
                        sources.append(player_idx[player])
                        targets.append(outcome_idx['Lost'])
                        values.append(loss_count)
                
                # Create color mapping
                node_colors = ['#3498db'] * len(roles) + ['#95a5a6'] * len(players) + ['#2ecc71', '#e74c3c']
                
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color=node_colors
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values
                    )
                )])
                
                fig.update_layout(
                    title_text="Player Decision Flow: Role → Player → Outcome",
                    font_size=12,
                    height=800
                )
                
                # Save as HTML file
                fig.write_html(str(output_path / 'player_decision_flow.html'))
                
                # Also save as static image
                fig.write_image(str(output_path / 'player_decision_flow.png'), width=1200, height=800)
                
            except ImportError as e:
                if 'kaleido' in str(e):
                    print("\nNote: Kaleido package is required for Sankey diagram static image export.")
                    print("Install it with: pip install kaleido")
                    
                    # Still try to save HTML if plotly is available and fig was created
                    if 'plotly.graph_objects' in sys.modules and fig is not None:
                        fig.write_html(str(output_path / 'player_decision_flow.html'))
                        print("Saved Sankey diagram as HTML only")
                else:
                    print(f"\nNote: Plotly is required for Sankey diagrams: {e}")
                    print("Install it with: pip install plotly")
                
        except Exception as e:
            print(f"Could not generate Sankey diagram: {e}")


def suggest_quality_criteria(df):
    """
    Suggest criteria for identifying high-quality data based on analysis.
    
    Args:
        df: pandas DataFrame
    """
    print("\nSuggested quality criteria:")
    
    criteria = []
    
    # Minimum message length
    if 'message_length' in df.columns:
        q75 = df['message_length'].quantile(0.75)
        criteria.append(f"- Message length greater than {int(q75)} characters")
    
    # Word count threshold
    if 'word_count' in df.columns:
        avg_word_count = df['word_count'].mean()
        criteria.append(f"- Word count greater than {int(avg_word_count)} words")
    
    # Role-specific criteria
    if all(col in df.columns for col in ['role', 'has_won_int']):
        for role in df['role'].unique():
            win_rate = df[df['role'] == role]['has_won_int'].mean()
            criteria.append(f"- For {role}s: win rate of {win_rate:.2f}")
    
    # Model-specific criteria
    if all(col in df.columns for col in ['llm_model', 'has_won_int']):
        models_by_win_rate = df.groupby('llm_model')['has_won_int'].mean().sort_values(ascending=False)
        top_model = models_by_win_rate.index[0]
        criteria.append(f"- Consider prioritizing {top_model} model (win rate: {models_by_win_rate.iloc[0]:.2f})")
    
    # Vote count criteria
    if 'vote_count' in df.columns and 'has_won_int' in df.columns:
        winning_avg_votes = df[df['has_won_int'] == 1]['vote_count'].mean()
        losing_avg_votes = df[df['has_won_int'] == 0]['vote_count'].mean()
        if winning_avg_votes < losing_avg_votes:
            criteria.append(f"- Lower vote count may correlate with winning (winning avg: {winning_avg_votes:.2f}, losing avg: {losing_avg_votes:.2f})")
        else:
            criteria.append(f"- Consider vote count patterns (winning avg: {winning_avg_votes:.2f}, losing avg: {losing_avg_votes:.2f})")
    
    # Message content criteria
    if 'discussion_message' in df.columns:
        criteria.append("- Messages that demonstrate strategic thinking")
        criteria.append("- Messages that reference specific game events")
    
    for criterion in criteria:
        print(criterion)


def main():
    """
    Main function to analyze the playthrough dataset.
    """
    parser = argparse.ArgumentParser(description='Analyze playthrough dataset for correlations and quality metrics.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='./out/analysis_output', help='Directory to save visualizations')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during analysis')
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
        
    try:
        # Load the dataset
        df = load_dataset(args.input)
        
        if df.empty:
            print("Error: Dataset is empty.")
            sys.exit(1)
            
        # Print column info if verbose
        if args.verbose:
            print("\nColumn info:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype} - {df[col].nunique()} unique values - {df[col].isna().sum()} missing values")
        
        # Perform analyses
        analyze_correlations(df)
        role_based_analysis(df)
        model_based_analysis(df)
        analyze_discussions(df)
        
        # Generate visualizations
        generate_visualizations(df, args.output_dir)
        
        # Suggest quality criteria
        suggest_quality_criteria(df)
        
        print(f"\nAnalysis complete. Visualizations saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
