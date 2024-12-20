import pandas as pd
import networkx as nx
from scipy import stats
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots



def load_dataframe(file_path: str, columns: list) -> pd.DataFrame:
    """Load a DataFrame from a file."""
    df = pd.read_csv(file_path, sep="\t", comment='#', header=None)
    df.columns = columns
    return df

def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values."""
    return df.dropna()

def drop_duplicates(df: pd.DataFrame, duplicate_columns) -> pd.DataFrame:
    """Drop duplicate rows."""
    return df.drop_duplicates(duplicate_columns)

def manage_categories(categories_df: pd.DataFrame) -> dict[str, list]:
    """Manage categories in the DataFrame."""
    categories_df["category"] = categories_df["category"].apply(lambda x: x.split(".")[1:])
    dict_categories = {}
    for index, row in categories_df.iterrows():
        dict_categories[row["page"]] = row["category"]
    return dict_categories


def clean_path(path: list) -> list:
    """Clean a path from back clicks."""
    clean_path = []
    count = 0
    for element in path[::-1]:
        if element == "<":
            count += 1
            continue
        if count > 0:
            count -= 1
            continue
        clean_path.append(element)
    return clean_path[::-1]

def manage_paths(paths_df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Manage paths in the DataFrame."""
    if "target" not in paths_df.columns:
        paths_df["rating"] = paths_df["rating"].fillna(-1)
    paths_df["path"] = paths_df["path"].apply(lambda x: x.split(";"))
    paths_df['nb_back_clicks'] = paths_df['path'].apply(lambda x: x.count('<'))
    paths_df['nb_clicks'] = paths_df["path"].apply(lambda x: len(x) - 1)
    paths_df['path_length'] = paths_df['nb_clicks'] - 2 * paths_df['nb_back_clicks']
    paths_df['source'] = paths_df["path"].apply(lambda x: x[0])
    if "target" not in paths_df.columns:
        paths_df['target'] = paths_df["path"].apply(lambda x: x[-1])
    else:
        paths_df["play_duration"] = paths_df["durationInSec"]
        paths_df.loc[paths_df["type"] == "timeout", "play_duration"] = paths_df.loc[
            paths_df["type"] == "timeout", "play_duration"].apply(lambda x: max(x - 1800, 0))

    paths_df["source_category"] = paths_df["source"].apply(lambda x: categories.get(x, None))
    paths_df["target_category"] = paths_df["target"].apply(lambda x: categories.get(x, None))
    paths_df = paths_df.dropna(subset=["source_category", "target_category"])
    paths_df["source_general_category"] = paths_df["source_category"].apply(lambda x: x[0])
    paths_df["target_general_category"] = paths_df["target_category"].apply(lambda x: x[0])

    paths_df["clean_path"] = paths_df["path"].apply(clean_path)

    return paths_df


def manage_links(links_df: pd.DataFrame) -> dict[str, list]:
    """Manage links in the DataFrame."""
    dict_links = {}
    for index, row in links_df.iterrows():
        if row["source"] not in dict_links:
            dict_links[row["source"]] = []
        dict_links[row["source"]].append(row["target"])
    return dict_links


def create_graph(links_df: pd.DataFrame) -> nx.DiGraph:
    """Create a graph from the links DataFrame."""
    graph = nx.from_pandas_edgelist(links_df,
        source="source", target="target", create_using=nx.DiGraph)
    return graph


def read_llm_paths(path: str) -> dict:
    """Read the LLM paths from a file."""
    with open(path, "r") as f:
        responses = json.load(f)
    return responses


def page_rank(graph, alpha=0.85):
    """Compute the PageRank of a graph."""
    return nx.pagerank(graph, alpha=alpha)

def path_length(list):
    """Compute the length of a path without the backclicks and removing first click"""
    return len(clean_path(list)) - 1

def paths_with_most_common_length(original_paths: list):
    """Returns the paths with the most common length"""
    most_common_length = pd.Series(original_paths).apply(len).mode().values[0]
    paths_with_most_common_length = [path for path in original_paths if len(path) == most_common_length]
    return paths_with_most_common_length

def mean_ranks(paths: list, ranks: dict):
    """Returns the mean ranks of the paths"""
    mean_ranks = []
    for i in range(len(paths[0])):
        rank_i = 0
        for path in paths:
            rank_i += ranks.get(path[i], 0)
        rank_i /= len(paths)
        mean_ranks.append(rank_i)
    return mean_ranks



def tstats_pvalues(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict):
    """Compute t-statistics and p-values to compare players and LLM paths

    Parameters:
        sources (list): sources list
        targets (list): targets list
        finished_paths_df (DataFrame): players finished paths
        llm_paths (dict): the LLM generated paths for each source-target pair   

    Returns:
        list: p-values
    """
    # Compute t-statistics and p-values to compare players and LLM paths
    t_stats = []
    p_values = []
    for source, target in zip(sources, targets):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        llm_source_target = llm_paths[source+"_"+target]
        
        # Add small noise to avoid identical values
        player_performance = player_paths["path_length"] + np.random.normal(0, 1e-10, len(player_paths))
        llm_performance = [len(path) for path in llm_source_target] + np.random.normal(0, 1e-10, len(llm_source_target))
        
        t_stat, p_value = stats.ttest_ind(llm_performance, player_performance, alternative='less')
        t_stats.append(abs(t_stat))
        p_values.append(p_value)
    
    return p_values


def compute_coordinates_per_step(links_coords_finished, links_coords_unfinished):
    """
    Computes the variance of x and y coordinates at each step of tha paths.
    
    Parameters:
        links_coords_finished (pd.DataFrame): DataFrame containing finished paths
        links_coords_unfinished (pd.DataFrame): DataFrame containing unfinished paths
    
    Returns:
    tuple: A tuple containing:
        - steps (list of int): List of step numbers from 1.
        - variance_x (list of float): List of variances of x coordinates for each step.
        - variance_y (list of float): List of variances of y coordinates for each step.
    """
    
    max_step = 7

    coords_per_step = {}
    for df in [links_coords_finished, links_coords_unfinished]:
        for _, line in df.iterrows():
            for i, coords in enumerate(line["normalized_links_coords"]):
                if i >= max_step:
                    break
                coords_per_step.setdefault(i, []).extend(coords)

    variance_x = []
    variance_y = []
    for step, coords in coords_per_step.items():
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        variance_x.append(np.var(x_coords))
        variance_y.append(np.var(y_coords))

    steps = list(range(1, max_step + 1))

    return steps, variance_x, variance_y
