import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



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
    
    fig = px.scatter(
        x=[f"{source} -> {target}" for source, target in zip(sources, targets)] * 2,
        y=player_means + llm_means,
        error_y=player_std_errors + llm_std_errors,
        labels={"x": "Source -> Target", "y": "Path Length"},
        title="Player Paths vs LLM mean path length per source-target pair",
        color=["Player Paths"] * len(player_means) + ["LLM Paths"] * len(llm_means)
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(len(sources))), ticktext=[f"{source} -> {target}" for source, target in zip(sources, targets)]),
        xaxis_title="Source -> Target",
        yaxis_title="Path Length",
        height=600
    )
    fig.show()

def plot_llm_vs_llm(sources: list, targets: list, llm1_paths: dict, llm2_paths: dict, model1_name: str, model2_name: str):
    """plots the means and std errors of the path lengths for two LLMs 

        sources (list): sources list
        targets (list): targets list
        llm1_paths (dict): the first LLM generated paths for each source-target pair
        llm2_paths (dict): the second LLM generated paths for each source-target pair
    """
    llm1_means = []
    llm1_std_errors = []
    llm2_means = []
    llm2_std_errors = []

    for source, target in zip(sources, targets):
        #compute llm1 paths length means
        llm1_source_target = llm1_paths[source+"_"+target]
        llm1_mean_length = 0
        for path in llm1_source_target:
            llm1_mean_length += len(path)
        llm1_mean_length = llm1_mean_length / len(llm1_source_target) if len(llm1_source_target) != 0 else 0
        #compute llm1 paths length standard errors
        llm1_std_error = 0
        for path in llm1_source_target:
            llm1_std_error += (len(path) - llm1_mean_length)**2
        llm1_std_error = llm1_std_error / len(llm1_source_target) if len(llm1_source_target) != 0 else 0
        llm1_std_error = llm1_std_error**0.5

        llm1_means.append(llm1_mean_length)
        llm1_std_errors.append(llm1_std_error)
        
        #compute llm paths length means 
        llm2_source_target = llm2_paths[source+"_"+target]
        llm2_mean_length = 0
        for path in llm2_source_target:
            llm2_mean_length += len(path)
        llm2_mean_length = llm2_mean_length / len(llm2_source_target) if len(llm2_source_target) != 0 else 0

        #compute llm paths length standard errors
        llm2_std_error = 0
        for path in llm2_source_target:
            llm2_std_error += (len(path) - llm2_mean_length)**2
        llm2_std_error = llm2_std_error / len(llm2_source_target) if len(llm2_source_target) != 0 else 0
        llm2_std_error = llm2_std_error**0.5

        llm2_means.append(llm2_mean_length)
        llm2_std_errors.append(llm2_std_error)
    
    fig = px.scatter(
        x=[f"{source} -> {target}" for source, target in zip(sources, targets)] * 2,
        y= llm1_means + llm2_means,
        error_y= llm1_std_errors + llm2_std_errors,
        labels={"x": "Source -> Target", "y": "Path Length"},
        title= model1_name +" VS " + model2_name + " mean path length per source-target pair",
        color=[model1_name] * len(llm1_means) + [model2_name] * len(llm2_means)
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(len(sources))), ticktext=[f"{source} -> {target}" for source, target in zip(sources, targets)]),
        xaxis_title="Source -> Target",
        yaxis_title="Path Length",
        height=600
    )
    fig.show()




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
    for source, target in zip(sources, targets):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        llm_source_target = llm_paths[source+"_"+target]
        
        # Add small noise to avoid identical values
        player_lengths = player_paths["path_length"] + np.random.normal(0, 1e-10, len(player_paths))
        llm_lengths = [len(path) for path in llm_source_target] + np.random.normal(0, 1e-10, len(llm_source_target))
        
        t_stat, p_value = stats.ttest_ind(llm_lengths, player_lengths, alternative='less')
        t_stats.append(abs(t_stat))
        p_values.append(p_value)  
    return p_values


def plot_tsatistics(sources: list, targets: list,p_values: list):
    """plot t-statistics and p-values"""
    fig = go.Figure()
    for i, source, target in zip(range(len(sources)), sources, targets):
        fig.add_trace(go.Bar(
            x=[f"{source} -> {target}"],
            y=[p_values[i]],
            name='p-values',
            showlegend=False,
            marker=dict(color='green')
        )
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(10)), 
            ticktext=[f"{source} -> {target}" for source, target in zip(sources, targets)],
            tickangle=45
        ),
        yaxis_title='p-values',
        title='p-values for Player Paths vs LLM Paths with corresponding p-values',
        height=600,
    )

    fig.show()

def plot_llm_vs_players_strategies(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict, ranks: dict):
    """plot the mean ranks for players against LLMs paths for each source-target pair for the most common path length"""
    llm_vs_players_ranks = []
    for idx, (source, target) in enumerate(zip(sources, targets)):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        
        most_common_length = player_paths["path_length"].mode().values[0]
        paths_with_most_common_length = player_paths[player_paths["path_length"] == most_common_length]

        mean_ranks = []
        for i in range(most_common_length + 1):
            rank_i = 0
            for path in paths_with_most_common_length["clean_path"]:
                rank_i += ranks[path[i]]
            rank_i /= len(paths_with_most_common_length)
            mean_ranks.append(rank_i)

        llm_paths_source_target = llm_paths[source + "_" + target]
        most_common_llm_path_length = max([len(path) for path in llm_paths_source_target])
        mean_ranks_llm = []
        for i in range(most_common_llm_path_length):
            rank_i = 0
            for path in llm_paths_source_target:
                if len(path) > i:
                    if path[i] not in ranks:
                        continue
                    rank_i += ranks[path[i]]
            rank_i /= len(llm_paths_source_target)
            mean_ranks_llm.append(rank_i)
        llm_vs_players_ranks.append({"source": source, "target": target, "player_ranks": mean_ranks, "llm_ranks": mean_ranks_llm})
    for ranks in llm_vs_players_ranks:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(ranks["player_ranks"]))),
            y=ranks["player_ranks"],
            mode='lines+markers',
            name='Player Ranks',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(ranks["llm_ranks"]))),
            y=ranks["llm_ranks"],
            mode='lines+markers',
            name='LLM Ranks',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f"Source: {ranks['source']} | Target: {ranks['target']}",
            xaxis_title="Step",
            yaxis_title="Mean Rank",
            legend_title="Legend"
        )

        fig.show()



