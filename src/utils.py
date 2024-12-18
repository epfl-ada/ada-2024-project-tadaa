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
    """plots the means and std errors of the path performance for humans players and LLMs 

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
    """plots the means and std errors of the path performance for two LLMs 

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
        player_performance = player_paths["path_length"] + np.random.normal(0, 1e-10, len(player_paths))
        llm_performance = [len(path) for path in llm_source_target] + np.random.normal(0, 1e-10, len(llm_source_target))
        
        t_stat, p_value = stats.ttest_ind(llm_performance, player_performance, alternative='less')
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


def compare_llms_and_prompts(compare_path_performance, multiply_metrics):
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


    if compare_path_performance or multiply_metrics:
        for key in llm_paths_qwen_detail_prompt:
            llm_paths_qwen_detail_prompt_performance[key] = [len(path) for path in llm_paths_qwen_detail_prompt.get(key, [])]
            llm_paths_llama_detail_prompt_performance[key] = [len(path) for path in llm_paths_llama_detail_prompt.get(key, [])]
            llm_paths_qwen_simple_prompt_performance[key] = [len(path) for path in llm_paths_qwen_simple_prompt.get(key, [])]
            llm_paths_llama_simple_prompt_performance[key] = [len(path) for path in llm_paths_llama_simple_prompt.get(key, [])]
        if multiply_metrics:
            for key in llm_paths_qwen_detail_prompt_performance:
                simple_qwen_frac = len(llm_paths_qwen_simple_prompt_performance[key])/100
                simple_llama_frac = len(llm_paths_llama_simple_prompt_performance[key])/100
                detail_qwen_frac = len(llm_paths_qwen_detail_prompt_performance[key])/30
                detail_llama_frac = len(llm_paths_llama_detail_prompt_performance[key])/30
                llm_paths_qwen_detail_prompt_performance[key] = [1/path_length*(detail_qwen_frac) for path_length in llm_paths_qwen_detail_prompt_performance[key]] 
                llm_paths_llama_detail_prompt_performance[key] = [1/path_length*(detail_llama_frac) for path_length in llm_paths_llama_detail_prompt_performance[key]]  
                llm_paths_qwen_simple_prompt_performance[key] = [1/path_length*(simple_qwen_frac) for path_length in llm_paths_qwen_simple_prompt_performance[key]] 
                llm_paths_llama_simple_prompt_performance[key] = [1/path_length*(simple_llama_frac) for path_length in llm_paths_llama_simple_prompt_performance[key]] 
    else:
        for key in llm_paths_qwen_detail_prompt:
            llm_paths_qwen_detail_prompt_performance[key] = len(llm_paths_qwen_detail_prompt.get(key, []))/30
            llm_paths_llama_detail_prompt_performance[key] = len(llm_paths_llama_detail_prompt.get(key, []))/30
            llm_paths_qwen_simple_prompt_performance[key] = len(llm_paths_qwen_simple_prompt.get(key, []))/100
            llm_paths_llama_simple_prompt_performance[key] = len(llm_paths_llama_simple_prompt.get(key, []))/100

    fig = go.Figure()

    for i, key in enumerate(llm_paths_qwen_detail_prompt_performance):
        if llm_paths_qwen_detail_prompt_performance[key]:
            qwen_detail_mean = np.mean(llm_paths_qwen_detail_prompt_performance[key])
            qwen_detail_std = np.std(llm_paths_qwen_detail_prompt_performance[key])
        else:
            qwen_detail_mean = 0
            qwen_detail_std = 0

        if llm_paths_llama_detail_prompt_performance[key]:
            llama_detail_mean = np.mean(llm_paths_llama_detail_prompt_performance[key])
            llama_detail_std = np.std(llm_paths_llama_detail_prompt_performance[key])
        else:
            llama_detail_mean = 0
            llama_detail_std = 0

        if llm_paths_qwen_simple_prompt_performance[key]:
            qwen_simple_mean = np.mean(llm_paths_qwen_simple_prompt_performance[key])
            qwen_simple_std = np.std(llm_paths_qwen_simple_prompt_performance[key])
        else:
            qwen_simple_mean = 0
            qwen_simple_std = 0

        if llm_paths_llama_simple_prompt_performance[key]:
            llama_simple_mean = np.mean(llm_paths_llama_simple_prompt_performance[key])
            llama_simple_std = np.std(llm_paths_llama_simple_prompt_performance[key])
        else:
            llama_simple_mean = 0
            llama_simple_std = 0

        fig.add_trace(go.Bar(
            x=[i - 0.3],
            y=[qwen_detail_mean],
            error_y=dict(type='data', array=[qwen_detail_std]),
            name='Qwen Detail Prompt',
            marker=dict(color='blue'),
            width=0.2
        ))

        fig.add_trace(go.Bar(
            x=[i - 0.1],
            y=[llama_detail_mean],
            error_y=dict(type='data', array=[llama_detail_std]),
            name='Llama Detail Prompt',
            marker=dict(color='red'),
            width=0.2
        ))

        fig.add_trace(go.Bar(
            x=[i + 0.1],
            y=[qwen_simple_mean],
            error_y=dict(type='data', array=[qwen_simple_std]),
            name='Qwen Simple Prompt',
            marker=dict(color='green'),
            width=0.2
        ))

        fig.add_trace(go.Bar(
            x=[i + 0.3],
            y=[llama_simple_mean],
            error_y=dict(type='data', array=[llama_simple_std]),
            name='Llama Simple Prompt',
            marker=dict(color='orange'),
            width=0.2
        ))
        fig.update_traces(showlegend=False)
        fig.data[0].showlegend = True
        fig.data[0].name = 'Qwen Detailed Prompt'
        fig.data[1].showlegend = True
        fig.data[1].name = 'Llama Detailed Prompt'
        fig.data[2].showlegend = True
        fig.data[2].name = 'Qwen Simple Prompt'
        fig.data[3].showlegend = True
        fig.data[3].name = 'Llama Simple Prompt'
    fig.update_layout(
        barmode='group',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(llm_paths_qwen_detail_prompt_performance))),
            ticktext=list(llm_paths_qwen_detail_prompt_performance.keys())
        ),
        xaxis_title='Source -> Target',
        yaxis_title='Performance Score',
        title='Comparison of LLM (success frequency / Paths performance) with Different Prompts and Models',
        height=600,
        bargap=0.15
    )

    fig.show()

def compare_llms_hub():
    with open("data/llm_responses_qwen_simple_prompt.json", "r") as f:
        llm_paths_qwen_simple_prompt = json.load(f)
    with open("data/llm_responses_qwen_simple_hub_prompt.json", "r") as f:
        llm_paths_qwen_simple_hub_prompt = json.load(f)
    with open("data/llm_responses_qwen_detailed_hub_prompt.json", "r") as f:
        llm_paths_qwen_detailed_hub_prompt = json.load(f)
    
    llm_paths_qwen_simple_prompt_performance = {}
    llm_paths_qwen_simple_hub_prompt_performance = {}
    llm_paths_qwen_detailed_hub_prompt_performance = {}

    for key in llm_paths_qwen_simple_prompt:
        llm_paths_qwen_simple_prompt_performance[key] = [len(path) for path in llm_paths_qwen_simple_prompt.get(key, [])]
        llm_paths_qwen_simple_hub_prompt_performance[key] = [len(path) for path in llm_paths_qwen_simple_hub_prompt.get(key, [])]
        llm_paths_qwen_detailed_hub_prompt_performance[key] = [len(path) for path in llm_paths_qwen_detailed_hub_prompt.get(key, [])]
        simple_qwen_frac = len(llm_paths_qwen_simple_prompt_performance[key])/100
        simple_hub_qwen_frac = len(llm_paths_qwen_simple_hub_prompt_performance[key])/30
        detailed_hub_qwen_frac = len(llm_paths_qwen_detailed_hub_prompt_performance[key])/30
        llm_paths_qwen_simple_prompt_performance[key] = [1/path_length*(simple_qwen_frac) for path_length in llm_paths_qwen_simple_prompt_performance[key]]
        llm_paths_qwen_simple_hub_prompt_performance[key] = [1/path_length*(simple_hub_qwen_frac) for path_length in llm_paths_qwen_simple_hub_prompt_performance[key]]
        llm_paths_qwen_detailed_hub_prompt_performance[key] = [1/path_length*(detailed_hub_qwen_frac) for path_length in llm_paths_qwen_detailed_hub_prompt_performance[key]]

    fig = go.Figure()

    for i, key in enumerate(llm_paths_qwen_simple_prompt_performance):
        if llm_paths_qwen_simple_prompt_performance[key]:
            simple_mean = np.mean(llm_paths_qwen_simple_prompt_performance[key])
            simple_std = np.std(llm_paths_qwen_simple_prompt_performance[key])
        else:
            simple_mean = 0
            simple_std = 0

        if llm_paths_qwen_simple_hub_prompt_performance[key]:
            simple_hub_mean = np.mean(llm_paths_qwen_simple_hub_prompt_performance[key])
            simple_hub_std = np.std(llm_paths_qwen_simple_hub_prompt_performance[key])
        else:
            simple_hub_mean = 0
            simple_hub_std = 0

        if llm_paths_qwen_detailed_hub_prompt_performance[key]:
            detailed_hub_mean = np.mean(llm_paths_qwen_detailed_hub_prompt_performance[key])
            detailed_hub_std = np.std(llm_paths_qwen_detailed_hub_prompt_performance[key])
        else:
            detailed_hub_mean = 0
            detailed_hub_std = 0

        fig.add_trace(go.Bar(
            x=[i - 0.2],
            y=[simple_mean],
            error_y=dict(type='data', array=[simple_std]),
            name='Qwen Simple Prompt',
            marker=dict(color='blue'),
            width=0.2
        ))

        fig.add_trace(go.Bar(
            x=[i],
            y=[simple_hub_mean],
            error_y=dict(type='data', array=[simple_hub_std]),
            name='Qwen Simple Hub Prompt',
            marker=dict(color='red'),
            width=0.2
        ))

        fig.add_trace(go.Bar(
            x=[i + 0.2],
            y=[detailed_hub_mean],
            error_y=dict(type='data', array=[detailed_hub_std]),
            name='Qwen Detailed Hub Prompt',
            marker=dict(color='green'),
            width=0.2
        ))
        fig.update_traces(showlegend=False)
        fig.data[0].showlegend = True
        fig.data[0].name = 'Qwen Simple Prompt'
        fig.data[1].showlegend = True
        fig.data[1].name = 'Qwen Simple Hub Prompt'
        fig.data[2].showlegend = True
        fig.data[2].name = 'Qwen Detailed Hub Prompt'
    fig.update_layout(
        barmode='group',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(llm_paths_qwen_simple_prompt_performance))),
            ticktext=list(llm_paths_qwen_simple_prompt_performance.keys())
        ),
        xaxis_title='Source -> Target',
        yaxis_title='Performance Score',
        title='Comparison of LLM (success frequency / Paths performance) with Different Prompts',
        height=600,
        bargap=0.15
    )

    fig.show()



def paths_with_most_common_length(original_paths: list):
    """Returns the paths with the most common length"""
    most_common_length = pd.Series(original_paths).apply(len).mode().values[0]
    print(most_common_length)
    paths_with_most_common_length = [path for path in original_paths if len(path) == most_common_length]
    return paths_with_most_common_length

def mean_ranks(paths: list, ranks: dict):
    """Returns the mean ranks of the paths"""
    mean_ranks = []
    print(paths)
    for i in range(len(paths[0])):
        rank_i = 0
        for path in paths:
            rank_i += ranks.get(path[i], 0)
        rank_i /= len(paths)
        mean_ranks.append(rank_i)
    return mean_ranks

def hub_llms_paths(ranks: dict):
    with open("data/llm_responses_qwen_simple_prompt_hub.json", "r") as f:
        llm_paths_qwen_simple_prompt_hub = json.load(f)
    with open("data/llm_responses_qwen_detailed_prompt_hub.json", "r") as f:
        llm_paths_qwen_detail_prompt_hub = json.load(f)
    with open("data/llm_responses_qwen_simple_prompt.json", "r") as f:
        llm_paths_qwen_simple_prompt = json.load(f)

    for key in llm_paths_qwen_simple_prompt:
        paths_qwen_simple_prompt = paths_with_most_common_length(llm_paths_qwen_simple_prompt[key])
        paths_qwen_simple_prompt_hub = paths_with_most_common_length(llm_paths_qwen_simple_prompt_hub[key])
        paths_qwen_detail_prompt_hub = paths_with_most_common_length(llm_paths_qwen_detail_prompt_hub[key])

        mean_ranks_qwen_simple_prompt = mean_ranks(paths_qwen_simple_prompt, ranks) 
        mean_ranks_qwen_simple_prompt_hub = mean_ranks(paths_qwen_simple_prompt_hub, ranks)
        mean_ranks_qwen_detail_prompt_hub = mean_ranks(paths_qwen_detail_prompt_hub, ranks)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(mean_ranks_qwen_simple_prompt))),
            y=mean_ranks_qwen_simple_prompt,
            mode='lines+markers',
            name='Qwen Simple Prompt',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(mean_ranks_qwen_simple_prompt_hub))),
            y=mean_ranks_qwen_simple_prompt_hub,
            mode='lines+markers',
            name='Qwen Simple Prompt Hub',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(mean_ranks_qwen_detail_prompt_hub))),
            y=mean_ranks_qwen_detail_prompt_hub,
            mode='lines+markers',
            name='Qwen Detail Prompt Hub',
            line=dict(color='green')
        ))
        fig.update_layout(
            title=f"Source: {key.split('_')[0]} | Target: {key.split('_')[1]}",
            xaxis_title="Step",
            yaxis_title="Mean Rank",
            legend_title="Legend"
        )

        fig.show()



def hub_impact():
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
        qwen_simple_frac = len(llm_paths_qwen_simple_prompt[key])/100
        qwen_simple_hub_frac = len(llm_paths_qwen_simple_prompt_hub[key])/30
        qwen_detail_hub_frac = len(llm_paths_qwen_detail_prompt_hub[key])/30
        llm_paths_qwen_simple_prompt_performance[key] = [qwen_simple_frac / len(path) for path in llm_paths_qwen_simple_prompt.get(key, [])]
        llm_paths_qwen_simple_prompt_hub_performance[key] = [qwen_simple_hub_frac / len(path) for path in llm_paths_qwen_simple_prompt_hub.get(key, [])]
        llm_paths_qwen_detail_prompt_hub_performance[key] = [qwen_detail_hub_frac / len(path) for path in llm_paths_qwen_detail_prompt_hub.get(key, [])]


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
    with open("data/llm_times_llama_simple_prompt.json", "r") as f:
        llm_times_llama_simple_prompt = json.load(f)
    with open("data/llm_times_llama_detailed_prompt.json", "r") as f:
        llm_times_qwen_simple_prompt = json.load(f)

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[time for time in llm_times_llama_simple_prompt],
        name='Llama Simple Prompt',
        marker=dict(color='blue')
    ))  
    fig.add_trace(go.Box(
        y=[time for time in llm_times_qwen_simple_prompt],
        name='Llama Detailed Prompt',
        marker=dict(color='red')
    ))
    fig.update_layout(
        title='Time Performance of Llama with Different Prompts',
        yaxis_title='Time (s)',
        height=600,
        yaxis_type="log"
    )
    fig.show()
