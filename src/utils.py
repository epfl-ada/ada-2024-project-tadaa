import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import json


def load_dataframe(file_path: str, columns: list) -> pd.DataFrame:
    """Load a DataFrame from a file."""
    df = pd.read_csv(file_path, sep="\t", comment='#')
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
    with open(path, "r") as f:
        responses = json.load(f)
    return responses


def page_rank(graph, alpha=0.85):
    return nx.pagerank(graph, alpha=alpha)

def path_length(list):
    """Compute the length of a path considering the backclicks and removing first click"""
    return len(clean_path(list)) - 1

def plot_llms_vs_players(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict):
    """plots the means and std errors of the path lengths for humans players and LLMs 

        sources (list): sources list
        targets (list): targets list
        finished_paths_df (DataFrame): players finished paths
        llm_paths (dict): the LLM generated paths for each source-target pair
    """
    llm_means = []
    llm_std_errors = []
    player_means = []
    player_std_errors = []

    for source, target in zip(sources, targets):
        # select paths with the same source and target
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        # compute players paths length means and standard errors
        player_mean_length = player_paths["path_length"].mean()
        player_std_error = player_paths["path_length"].sem()

        player_std_errors.append(player_std_error)
        player_means.append(player_mean_length)
        
        #compute llm paths length means 
        llm_source_target = llm_paths[source+"_"+target]
        llm_mean_length = 0
        for path in llm_source_target:
            llm_mean_length += len(path)
        llm_mean_length /= len(llm_source_target)
        #compute llm paths length standard errors
        llm_std_error = 0
        for path in llm_source_target:
            llm_std_error += (len(path) - llm_mean_length)**2
        llm_std_error /= len(llm_source_target)
        llm_std_error = llm_std_error**0.5

        llm_means.append(llm_mean_length)
        llm_std_errors.append(llm_std_error)
    
    fig, ax = plt.subplots()
    ax.errorbar(range(10), player_means, yerr=player_std_errors, fmt="o", label="Player Paths")
    ax.errorbar(range(10), llm_means, yerr=llm_std_errors, fmt="o", label="LLM Paths")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{source} -> {target}" for source, target in zip(sources, targets)], rotation=90)  
    ax.set_ylabel("Path Length")
    ax.set_xlabel("Source -> Target")
    plt.title("Player Paths vs LLM mean path length per source-target pair")
    ax.legend()
    plt.show()


def tstats_pvalues(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict):
    """Compute t-statistics and p-values to compare players and LLM paths

        sources (list): sources list
        targets (list): targets list
        finished_paths_df (DataFrame): players finished paths
        llm_paths (dict): the LLM generated paths for each source-target pair   
    """
    # Compute t-statistics and p-values to compare players and LLM paths
    t_stats = []
    p_values = []
    t_critical = []
    for source, target in zip(sources, targets):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        llm_source_target = llm_paths[source+"_"+target]
        
        # Add small noise to avoid identical values
        player_lengths = player_paths["path_length"] + np.random.normal(0, 1e-10, len(player_paths))
        llm_lengths = [len(path) for path in llm_source_target] + np.random.normal(0, 1e-10, len(llm_source_target))
        
        t_stat, p_value = stats.ttest_ind(player_lengths, llm_lengths)
        t_stats.append(abs(t_stat))
        p_values.append(p_value)  
        t_critical.append(stats.t.ppf(1-0.05/2, len(player_paths["path_length"]) + len(llm_source_target) - 2))
    return t_stats, p_values, t_critical