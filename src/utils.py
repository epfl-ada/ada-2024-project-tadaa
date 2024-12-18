import pandas as pd
import networkx as nx
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

def plot_llms_vs_players(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict):
    """Plot the means and std errors of the average path lengths for human players and LLMs 

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
        # Select paths with the same source and target
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        # Compute players path length means and standard errors
        player_mean_length = player_paths["path_length"].mean()
        player_std_error = player_paths["path_length"].sem()

        player_std_errors.append(player_std_error)
        player_means.append(player_mean_length)
        
        # Compute llm path length means 
        llm_source_target = llm_paths[source+"_"+target]
        llm_mean_length = 0
        for path in llm_source_target:
            llm_mean_length += len(path)
        llm_mean_length /= len(llm_source_target)

        # Compute llm paths length standard errors
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
        title="Player vs LLM Mean Path Length per source-target pair",
        color=["Player Paths"] * len(player_means) + ["LLM Paths"] * len(llm_means)
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
        player_performance = player_paths["path_length"] + np.random.normal(0, 1e-10, len(player_paths))
        llm_performance = [len(path) for path in llm_source_target] + np.random.normal(0, 1e-10, len(llm_source_target))
        
        t_stat, p_value = stats.ttest_ind(llm_performance, player_performance, alternative='less')
        t_stats.append(abs(t_stat))
        p_values.append(p_value)  
    return p_values


def plot_tsatistics(sources: list, targets: list,p_values: list):
    """Plot t-statistics and p-values"""
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
        title='P-values for Player vs LLM Mean Path Length',
        height=600,
    )

    fig.show()

def plot_llm_vs_players_strategies(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict, ranks: dict):
    """Plot the mean ranks for players against LLMs paths for each source-target pair for the most common path length"""
    llm_vs_players_ranks = []
    for _, (source, target) in enumerate(zip(sources, targets)):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        # Compute the mean ranks for players
        mean_ranks_players = mean_ranks(list(paths_with_most_common_length(list(player_paths["clean_path"]))), ranks)

        llm_paths_source_target = llm_paths[source + "_" + target]
        # Compute the mean ranks for LLMs
        mean_ranks_llm = mean_ranks(paths_with_most_common_length(llm_paths_source_target), ranks)

        llm_vs_players_ranks.append({"source": source, "target": target, "player_ranks": mean_ranks_players, "llm_ranks": mean_ranks_llm})

    # Plot the mean ranks for players against LLMs paths for each source-target pair
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
            xaxis_title="Step in the path",
            yaxis_title="Mean Rank",
        )

        fig.show()


def compare_llms_and_prompts():
    """Compare the performance of two different LLMs with two different prompts"""

    with open("data/llm_responses_llama_simple_prompt.json", "r") as f:
        llm_paths_llama_simple_prompt = json.load(f)
    with open("data/llm_responses_qwen_simple_prompt.json", "r") as f:
        llm_paths_qwen_simple_prompt = json.load(f)
    with open("data/llm_responses_qwen_detailed_prompt.json", "r") as f:
        llm_paths_qwen_detail_prompt = json.load(f)
    with open("data/llm_responses_llama_detailed_prompt.json", "r") as f:     
        llm_paths_llama_detail_prompt = json.load(f)

    llm_paths_llama_detail_prompt_performance = {}
    llm_paths_qwen_detail_prompt_performance = {}
    llm_paths_llama_simple_prompt_performance = {}
    llm_paths_qwen_simple_prompt_performance = {}

    for key in llm_paths_qwen_simple_prompt:
        simple_qwen_frac = len(llm_paths_qwen_simple_prompt.get(key,[]))/100
        simple_llama_frac = len(llm_paths_llama_simple_prompt.get(key,[]))/100
        detail_qwen_frac = len(llm_paths_qwen_detail_prompt.get(key,[]))/30
        detail_llama_frac = len(llm_paths_llama_detail_prompt.get(key,[]))/30
        llm_paths_qwen_detail_prompt_performance[key] = [detail_qwen_frac/len(path) for path in llm_paths_qwen_detail_prompt.get(key, [])]
        llm_paths_llama_detail_prompt_performance[key] = [detail_llama_frac/len(path) for path in llm_paths_llama_detail_prompt.get(key, [])]
        llm_paths_qwen_simple_prompt_performance[key] = [simple_qwen_frac/len(path) for path in llm_paths_qwen_simple_prompt.get(key, [])]
        llm_paths_llama_simple_prompt_performance[key] = [simple_llama_frac/len(path) for path in llm_paths_llama_simple_prompt.get(key, [])]

    fig = go.Figure()
    def check_empty(paths_list):
        if len(paths_list) > 0:
            return np.mean(paths_list), np.std(paths_list)
        else:
            return 0, 0
    for i, key in enumerate(llm_paths_qwen_detail_prompt_performance):
        qwen_detail_mean, qwen_detail_std = check_empty(llm_paths_qwen_detail_prompt_performance[key])
        llama_detail_mean, llama_detail_std = check_empty(llm_paths_llama_detail_prompt_performance[key])
        qwen_simple_mean, qwen_simple_std = check_empty(llm_paths_qwen_simple_prompt_performance[key])
        llama_simple_mean, llama_simple_std = check_empty(llm_paths_llama_simple_prompt_performance[key])
                                                
        fig.add_trace(go.Bar(
            x=[key],
            y=[qwen_detail_mean],
            error_y=dict(type='data', array=[qwen_detail_std]),
            name='Qwen Detail Prompt',
            marker=dict(color='blue'),
            width=0.2,
            offsetgroup=0,
            showlegend=(i == 0)  

        ))

        fig.add_trace(go.Bar(
            x=[key],
            y=[llama_detail_mean],
            error_y=dict(type='data', array=[llama_detail_std]),
            name='Llama Detail Prompt',
            marker=dict(color='red'),
            width=0.2,
            offsetgroup=1,
            showlegend=(i == 0)  

        ))

        fig.add_trace(go.Bar(
            x=[key],
            y=[qwen_simple_mean],
            error_y=dict(type='data', array=[qwen_simple_std]),
            name='Qwen Simple Prompt',
            marker=dict(color='green'),
            width=0.2,
            offsetgroup=2,
            showlegend=(i == 0)  

        ))

        fig.add_trace(go.Bar(
            x=[key],
            y=[llama_simple_mean],
            error_y=dict(type='data', array=[llama_simple_std]),
            name='Llama Simple Prompt',
            marker=dict(color='orange'),
            width=0.2,
            offsetgroup=3,
            showlegend=(i == 0) 
        ))

        fig.update_layout(
            barmode='group',
            xaxis_title='Source -> Target',
            yaxis_title='Performance Score',
            title='Comparison of LLM Performance Score with Different Prompts and Models',
            height=700,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.5,
                xanchor="center",
                x=0.5
            )
        )

    fig.show()


def hub_impact():
    """Plot the impact of the hub on the performance of the LLMs"""

    with open("data/llm_responses_qwen_simple_hub_prompt.json", "r") as f:
        llm_paths_qwen_simple_prompt_hub = json.load(f)
    with open("data/llm_responses_qwen_detailed_hub_prompt.json", "r") as f:
        llm_paths_qwen_detail_prompt_hub = json.load(f)
    with open("data/llm_responses_qwen_simple_prompt.json", "r") as f:
        llm_paths_qwen_simple_prompt = json.load(f)

    llm_paths_qwen_simple_prompt_performance = {}
    llm_paths_qwen_simple_prompt_hub_performance = {}
    llm_paths_qwen_detail_prompt_hub_performance = {}

    for key in llm_paths_qwen_simple_prompt:
        # Compute the success frequency of the LLMs
        qwen_simple_frac = len(llm_paths_qwen_simple_prompt[key])/100
        qwen_simple_hub_frac = len(llm_paths_qwen_simple_prompt_hub[key])/30
        qwen_detail_hub_frac = len(llm_paths_qwen_detail_prompt_hub[key])/30
        # Compute the performance score of the LLMs (success frequency / Paths performance)
        llm_paths_qwen_simple_prompt_performance[key] = [qwen_simple_frac / len(path) for path in llm_paths_qwen_simple_prompt.get(key, [])]
        llm_paths_qwen_simple_prompt_hub_performance[key] = [qwen_simple_hub_frac / len(path) for path in llm_paths_qwen_simple_prompt_hub.get(key, [])]
        llm_paths_qwen_detail_prompt_hub_performance[key] = [qwen_detail_hub_frac / len(path) for path in llm_paths_qwen_detail_prompt_hub.get(key, [])]

    # Compute the average performance score of the LLMs
    llm_paths_qwen_simple_prompt_averages = [sum(llm_paths_qwen_simple_prompt_performance[key])/len(llm_paths_qwen_simple_prompt_performance[key]) for key in llm_paths_qwen_simple_prompt_performance] 
    llm_paths_qwen_simple_prompt_hub_averages = [sum(llm_paths_qwen_simple_prompt_hub_performance[key])/len(llm_paths_qwen_simple_prompt_hub_performance[key]) for key in llm_paths_qwen_simple_prompt_hub_performance] 
    llm_paths_qwen_detail_prompt_hub_averages = [sum(llm_paths_qwen_detail_prompt_hub_performance[key])/len(llm_paths_qwen_detail_prompt_hub_performance[key]) for key in llm_paths_qwen_detail_prompt_hub_performance] 

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = ["Qwen Simple Prompt"],
        y = [np.mean(llm_paths_qwen_simple_prompt_averages)],
        error_y=dict(type='data', array=[np.std(llm_paths_qwen_simple_prompt_averages)]),
        name='Qwen Simple Prompt',
        marker=dict(color='blue'),
        width=0.2
    ))

    fig.add_trace(go.Bar(
        x = ["Qwen Simple Prompt Hub"],
        y = [np.mean(llm_paths_qwen_simple_prompt_hub_averages)],
        error_y=dict(type='data', array=[np.std(llm_paths_qwen_simple_prompt_hub_averages)]),
        name='Qwen Simple Prompt Hub',
        marker=dict(color='red'),
        width=0.2
    ))

    fig.add_trace(go.Bar(
        x = ["Qwen Detail Prompt Hub"],
        y = [np.mean(llm_paths_qwen_detail_prompt_hub_averages)],
        error_y=dict(type='data', array=[np.std(llm_paths_qwen_detail_prompt_hub_averages)]),
        name='Qwen Detail Prompt Hub',
        marker=dict(color='green'),
        width=0.2
    ))

    fig.update_traces(showlegend=False)
    fig.data[0].showlegend = True
    fig.data[0].name = 'Qwen Simple Prompt'
    fig.data[1].showlegend = True
    fig.data[1].name = 'Qwen Simple Prompt Hub'
    fig.data[2].showlegend = True
    fig.data[2].name = 'Qwen Detail Prompt Hub'
    fig.update_layout(
        barmode='group',
        xaxis_title='Source -> Target',
        yaxis_title='Performance Score',
        title='Comparison of Qwen Performance with Different Prompts',
        height=600,
        bargap=0.15
    )

    fig.show()


def average_ranks_first_step(rank: dict):
    llm_paths_qwen_simple_prompt = read_llm_paths("data/llm_responses_qwen_simple_prompt.json")
    llm_paths_qwen_detail_prompt_hub = read_llm_paths("data/llm_responses_qwen_detailed_hub_prompt.json")
    llm_paths_qwen_simple_prompt_hub = read_llm_paths("data/llm_responses_qwen_simple_hub_prompt.json")

    average_ranks_first_step_qwen_simple_prompt = []
    average_ranks_first_step_qwen_detail_prompt_hub = []
    average_ranks_first_step_qwen_simple_prompt_hub = []

    for key in llm_paths_qwen_simple_prompt:
        for path in llm_paths_qwen_simple_prompt[key]:
            if path[1] not in rank:
                continue
            average_ranks_first_step_qwen_simple_prompt.append(rank.get(path[1], 0))
    for key in llm_paths_qwen_detail_prompt_hub:
        for path in llm_paths_qwen_detail_prompt_hub[key]:
            if path[1] not in rank:
                continue
            average_ranks_first_step_qwen_detail_prompt_hub.append(rank.get(path[1], 0))
    for key in llm_paths_qwen_simple_prompt_hub:
        for path in llm_paths_qwen_simple_prompt_hub[key]:
            if path[1] not in rank:
                continue
            average_ranks_first_step_qwen_simple_prompt_hub.append(rank.get(path[1], 0))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Qwen Simple Prompt"],
        y=[np.mean(average_ranks_first_step_qwen_simple_prompt)],
        error_y=dict(type='data', array=[np.std(average_ranks_first_step_qwen_simple_prompt)]),
        name='Qwen Simple Prompt',
        marker=dict(color='blue'),
        width=0.2
    ))

    fig.add_trace(go.Bar(
        x=["Qwen Simple Prompt Hub"],
        y=[np.mean(average_ranks_first_step_qwen_simple_prompt_hub)],
        error_y=dict(type='data', array=[np.std(average_ranks_first_step_qwen_simple_prompt_hub)]),
        name='Qwen Simple Prompt Hub',
        marker=dict(color='red'),
        width=0.2
    ))

    fig.add_trace(go.Bar(
        x=["Qwen Detail Prompt Hub"],
        y=[np.mean(average_ranks_first_step_qwen_detail_prompt_hub)],
        error_y=dict(type='data', array=[np.std(average_ranks_first_step_qwen_detail_prompt_hub)]),
        name='Qwen Detail Prompt Hub',
        marker=dict(color='green'),
        width=0.2
    ))

    fig.update_traces(showlegend=False)
    fig.data[0].showlegend = True
    fig.data[0].name = 'Qwen Simple Prompt'
    fig.data[1].showlegend = True
    fig.data[1].name = 'Qwen Simple Prompt Hub'
    fig.data[2].showlegend = True
    fig.data[2].name = 'Qwen Detailed Prompt Hub'
    fig.update_layout(
        barmode='group',
        xaxis_title='Source -> Target',
        yaxis_title='Mean Rank',
        title='Comparison of ranks of the first step of LLM Paths with Different Prompts',
        height=600,
        bargap=0.15
    )

    fig.show()


def plot_llm_times():
    """Plot the time performance of Llama with different prompts"""

    with open("data/llm_times_llama_simple_prompt.json", "r") as f:
        llm_times_llama_simple_prompt = json.load(f)
    with open("data/llm_times_llama_detailed_prompt.json", "r") as f:
        llm_times_qwen_simple_prompt = json.load(f)

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[time for time in llm_times_llama_simple_prompt if time>0],
        name='Llama Simple Prompt',
        marker=dict(color='blue')
    ))  
    fig.add_trace(go.Box(
        y=[time for time in llm_times_qwen_simple_prompt if time>0],
        name='Llama Detailed Prompt',
        marker=dict(color='red')
    ))
    fig.update_layout(
        title='Time Performance of Llama with Different Prompts',
        yaxis_title='Time in log scale (s)',
        height=600,
        yaxis_type="log"
    )
    fig.show()
