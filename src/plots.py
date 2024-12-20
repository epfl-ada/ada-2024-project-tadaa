import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipysigma import Sigma
from .utils import read_llm_paths, mean_ranks, paths_with_most_common_length, compute_coordinates_per_step
from src.crowd import find_defeated_crowd, create_graph
from src.crowd import find_defeated_crowd, create_graph



def plot_tsatistics(sources: list, targets: list,p_values: list, download=True):
    """Plot t-statistics and p-values

    Args:
        sources (list): sources list
        targets (list): targets list
        p_values (list): p-values
    """
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
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()
    if download:
        fig.write_html("p_values.html")

def plot_llms_vs_players(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict, download=True):
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
        color=["Player"] * len(player_means) + ["LLM"] * len(llm_means),
        color_discrete_map={"Player": "blue", "LLM": "red"}
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(len(sources))), ticktext=[f"{source} -> {target}" for source, target in zip(sources, targets)]),
        xaxis_title="Source -> Target",
        yaxis_title="Path Length",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            title=""
        )
    )
    fig.show()

    if download:
        fig.write_html("llm_vs_players.html")
    
def plot_llm_frequency(sources: list, targets: list,llm_paths: dict, download=True):
    number_of_finished_paths = {
    "Source_Target": [f"{source} -> {target}" for source, target in zip(sources, targets)],
    "Number_of_LLM_paths": [len(llm_paths[f"{source}_{target}"]) for source, target in zip(sources, targets)]
    }
    fig = px.bar(number_of_finished_paths, x="Source_Target", y="Number_of_LLM_paths", 
             title="Number of LLM paths finished by Qwen Simple Prompt out of 100 for each source-target pair")

    fig.update_layout(
    xaxis_title="Source -> Target",
    yaxis_title="Number of LLM paths",
    xaxis_tickangle=45
    )
    fig.show()

    if download:
        fig.write_html("llm_frequency.html")


def plot_llm_vs_players_strategies(sources: list, targets: list, finished_paths_df: pd.DataFrame, llm_paths: dict, ranks: dict, download=True):
    """Plot the mean ranks for players against
    LLMs paths for each source-target pair for the most common path length


    Args:
        sources (list): the list of sources
        targets (list): the list of targets
        finished_paths_df (pd.DataFrame): the dataframe containing the finished paths
        llm_paths (dict): the LLM generated paths for each source-target pair
        ranks (dict): the ranks of the paths
    """
    llm_vs_players_ranks = []
    for _, (source, target) in enumerate(zip(sources, targets)):
        player_paths = finished_paths_df[(finished_paths_df["source"] == source) & (finished_paths_df["target"] == target)]
        # Compute the mean ranks for players
        mean_ranks_players = mean_ranks(list(paths_with_most_common_length(list(player_paths["clean_path"]))), ranks)

        llm_paths_source_target = llm_paths[source + "_" + target]
        # Compute the mean ranks for LLMs
        mean_ranks_llm = mean_ranks(paths_with_most_common_length(llm_paths_source_target), ranks)

        llm_vs_players_ranks.append({"source": source, "target": target, "player_ranks": mean_ranks_players, "llm_ranks": mean_ranks_llm})

    subplot_titles = [f"Source: {ranks['source']} | Target: {ranks['target']}" for ranks in llm_vs_players_ranks]   
    last = subplot_titles.pop()
    subplot_titles.append("")
    subplot_titles.append(last)
    
    fig = make_subplots(
        rows=4, cols=3, subplot_titles=subplot_titles)
    for i, title in enumerate(fig['layout']['annotations']):
        title['font'] = dict(size=14)
    for i, ranks in enumerate(llm_vs_players_ranks):
        row, col = (4, 2) if i == 9 else (i // 3 + 1, i % 3 + 1)
        
        for rank_type, color in [("player_ranks", 'blue'), ("llm_ranks", 'red')]:
            fig.add_trace(go.Scatter(
                x=list(range(len(ranks[rank_type]))),
                y=ranks[rank_type],
                mode='lines+markers',
                name=f'{rank_type.replace("_", " ").title()}',
                line=dict(color=color),
                showlegend=(i == 0)  # Show legend only for the first subplot
            ), row=row, col=col)
        
    fig.update_xaxes(title_text="Step", row=row, col=col)
    fig.update_yaxes(title_text="Mean Rank", row=row, col=col)

    fig.update_layout(
        width=1000,
        height=900,
        title_text="Mean Ranks for Players vs LLMs Paths",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()


    if download:
        fig.write_html("llm_vs_players_strategies.html")
    

def average_ranks_first_step(rank: dict, download=True):
    """Plot the average ranks of the first step of LLM Paths with Different Prompts

    Args:
        rank (dict): the rank of the first step of the LLM paths
    """
    llm_paths_qwen_simple_prompt = read_llm_paths("data/llm_responses_qwen_simple_prompt.json")
    llm_paths_qwen_detail_prompt_hub = read_llm_paths("data/llm_responses_qwen_detailed_hub_prompt.json")
    llm_paths_qwen_simple_prompt_hub = read_llm_paths("data/llm_responses_qwen_simple_hub_prompt.json")

    average_ranks = {
        "simple": [],
        "detail_hub": [],
        "simple_hub": []
    }

    for prompt, paths in {
        "simple": llm_paths_qwen_simple_prompt,
        "detail_hub": llm_paths_qwen_detail_prompt_hub,
        "simple_hub": llm_paths_qwen_simple_prompt_hub
    }.items():
        for key in paths:
            for path in paths[key]:
                if path[1] in rank:
                    average_ranks[prompt].append(rank.get(path[1], 0))

    fig = go.Figure()

    for name, color, values in [
        ("Qwen Simple Prompt", 'blue', average_ranks["simple"]),
        ("Qwen Simple Prompt Hub", 'red', average_ranks["simple_hub"]),
        ("Qwen Detail Prompt Hub", 'green', average_ranks["detail_hub"])
    ]:
        fig.add_trace(go.Bar(
            x=[name],
            y=[np.mean(values)],
            error_y=dict(type='data', array=[np.std(values)]),
            name=name,
            marker=dict(color=color),
            width=0.2
        ))

    fig.update_traces(showlegend=False)
    for i in range(3):
        fig.data[i].showlegend = True

    fig.update_layout(
        barmode='group',
        xaxis_title='Source -> Target',
        yaxis_title='Mean Rank',
        title='Comparison of ranks of the first step of LLM Paths with Different Prompts',
        height=600,
        bargap=0.15,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )

    )

    fig.show()

    if download:
        fig.write_html("average_ranks_first_step.html")



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


def plot_top_game_pairs(finished_paths: pd.DataFrame, number_of_pairs: int):
    """Plot the top game pairs"""

    top_pairs = finished_paths.groupby(["source", "target"]).size().reset_index(name="count").sort_values("count", ascending=False).head(number_of_pairs)
    top_pairs_count = top_pairs["count"].sum()
    total_games = finished_paths.shape[0]
    top_pairs["source_target"] = top_pairs["source"] + " -> " + top_pairs["target"]
    counts = top_pairs["count"].tolist()
    fig = px.bar(counts, orientation='h', title=f"Top {number_of_pairs} pairs of source and target", height=700, width = 1200)
    fig.update_traces(
        hovertemplate='source_target=%{text}<extra></extra>, count=%{x}',
        text=top_pairs["source_target"], textposition='outside')
    fig.update_layout(
        xaxis_title="Count",
        yaxis=dict(showticklabels=False), 
        yaxis_title="", 
        xaxis_type="log", 
        showlegend=False,
        width = 1000
    )
    fig.show()
    print(f"Top {number_of_pairs} pairs represent {top_pairs_count/total_games*100:.2f}% of the total games")

def source_target_distribution(finished_paths: pd.DataFrame, number_of_pairs: int, categories: dict):
    """Plot the distribution of the source and target categories"""

    top_pairs = finished_paths.groupby(["source", "target"]).size().reset_index(name="count").sort_values("count", ascending=False) 
    top_pairs["source_category"] = top_pairs["source"].apply(lambda x: categories.get(x, None)[0]) 
    top_pairs["target_category"] = top_pairs["target"].apply(lambda x: categories.get(x, None)[0])
    finished_paths_st = top_pairs.groupby(['source_category', 'target_category']).size().reset_index(name='count').sort_values('count', ascending=False)

    top_pairs_st = top_pairs.head(number_of_pairs).groupby(['source_category', 'target_category']).size().reset_index(name='count').sort_values('count', ascending=False)
    category_colors = {
        'Religion': 'rgb(47,84,133)',
        'History': 'rgb(208,195,39)',
        'Geography': 'rgb(61,193,195)',
        'Mathematics': 'rgb(158,54,135)',
        'Everyday_life': 'rgba(106,189,67,255)',
        'People': 'rgba(247,235,20,255)',
        'Philosophy': 'rgb(128, 0, 128)',
        'Reference': 'rgb(233,30,37)',
        'Science': 'rgba(55,83,166,255)',
        'Countries': 'rgb(255, 192, 203)',
        'Language_and_literature': 'rgba(251,163,25,255)',
        'Design_and_Technology': 'rgb(0, 128, 128)',
        'Art': 'rgb(0, 255, 255)',
        'IT': 'rgb(255, 215, 0)',
        "Citizenship": 'rgb(255, 0, 0)',
        "Music": 'rgb(0, 0, 255)',
        "Business_Studies": 'rgb(0, 255, 0)',
    }
    def create_sankey_plot(df, title):
        return go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=df['source_category'].tolist() + df['target_category'].tolist(),
                color=[category_colors[cat] for cat in df['source_category'].tolist() + df['target_category'].tolist()]
            ),
            link=dict(
                source=[df['source_category'].tolist().index(cat) for cat in df['source_category']],
                target=[len(df['source_category'].tolist()) + df['target_category'].tolist().index(cat) for cat in df['target_category']],
                value=df['count'],
                color=[category_colors[cat] for cat in df['source_category']]
            )
        )]).update_layout(title_text=title, font_size=10)

    fig_alluvial1 = create_sankey_plot(top_pairs_st, "Source to target flows of top pairs")
    fig_alluvial2 = create_sankey_plot(finished_paths_st, "Source to target flows of all pairs")
    fig_combined = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Source to target flows of top {number_of_pairs} pairs", "Source to target flows of all pairs"),
        specs=[[{'type': 'sankey'}, {'type': 'sankey'}]]
    )

    for trace in fig_alluvial1.data + fig_alluvial2.data:
        fig_combined.add_trace(trace, row=1, col=1 if trace in fig_alluvial1.data else 2)

    fig_combined.update_layout(height=600, width=1000)
    fig_combined.show()


def compare_llms_and_prompts(download=True):
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
        means = [
            check_empty(llm_paths_qwen_detail_prompt_performance[key]),
            check_empty(llm_paths_llama_detail_prompt_performance[key]),
            check_empty(llm_paths_qwen_simple_prompt_performance[key]),
            check_empty(llm_paths_llama_simple_prompt_performance[key])
        ]
        colors = ['blue', 'red', 'green', 'orange']
        names = ['Qwen Detail Prompt', 'Llama Detail Prompt', 'Qwen Simple Prompt', 'Llama Simple Prompt']
        offsetgroups = range(4)
        
        for j, (mean, color, name, offset) in enumerate(zip(means, colors, names, offsetgroups)):
            fig.add_trace(go.Bar(
                x=[key],
                y=[mean[0]],
                error_y=dict(type='data', array=[mean[1]]),
                name=name,
                marker=dict(color=color),
                width=0.2,
                offsetgroup=offset,
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

    if download:
        fig.write_html("llm_performance.html")


def hub_impact(download=True):
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

    data = [
        ("Qwen Simple Prompt", llm_paths_qwen_simple_prompt_averages, 'blue'),
        ("Qwen Simple Prompt Hub", llm_paths_qwen_simple_prompt_hub_averages, 'red'),
        ("Qwen Detail Prompt Hub", llm_paths_qwen_detail_prompt_hub_averages, 'green')
    ]

    for i, (name, values, color) in enumerate(data):
        fig.add_trace(go.Bar(
            x=[name],
            y=[np.mean(values)],
            error_y=dict(type='data', array=[np.std(values)]),
            name=name,
            marker=dict(color=color),
            width=0.2
        ))

    fig.update_traces(showlegend=False)
    for i in range(len(data)):
        fig.data[i].showlegend = True

    fig.update_layout(
        barmode='group',
        xaxis_title='Source -> Target',
        yaxis_title='Performance Score',
        title='Comparison of Qwen Performance with Different Prompts with and without Hub',
        height=600,
        bargap=0.15,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()

    if download:
        fig.write_html("hub_performance.html")


def plot_coordinates_distribution(all_links_coords, title, download_path=None):
    """Plot the distribution of the coordinates of the links between articles in the path"""
    x_coords = []
    y_coords = []
    for coords in all_links_coords:
        x_coords.append(coords[0])
        y_coords.append(coords[1])

    fig = px.density_contour(
        x=x_coords, y=y_coords, 
        title=title, 
        labels={'x': 'X Coordinates', 'y': 'Y Coordinates'},
        histfunc="count", 
    )
    fig.update_traces(contours_coloring="fill", colorscale="Reds")
    fig.update_yaxes(range=[4000, 0])
    fig.update_xaxes(range=[0, 1920])
    fig.update_layout(
        height=600
    )
    fig.show()

    if download_path:
        fig.write_html(download_path)


def compare_coordinates_distribution(coords1, coords2, title1, title2, download_path=None):
    """Compare two distributions of the coordinates of the links between articles in the path"""
    x_coords1, y_coords1 = zip(*coords1)
    x_coords2, y_coords2 = zip(*coords2)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2))

    fig.add_trace(
        go.Histogram2dContour(
            x=x_coords1, y=y_coords1,
            colorscale='Reds',
            contours_coloring='fill',
            showscale=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram2dContour(
            x=x_coords2, y=y_coords2,
            colorscale='Reds',
            contours_coloring='fill',
            showscale=False
        ),
        row=1, col=2
    )

    fig.update_yaxes(range=[4000, 0], row=1, col=1)
    fig.update_xaxes(range=[0, 1920], row=1, col=1)
    fig.update_yaxes(range=[4000, 0], row=1, col=2)
    fig.update_xaxes(range=[0, 1920], row=1, col=2)

    fig.update_layout(
        height=600,
        title_text="Distribution of links coordinates of the unfinished and finished paths",
    )

    fig.show()

    if download_path:
        fig.write_html(download_path)

def plot_link_coordinates_variance_per_step(links_coords_finished, links_coords_unfinished, download_path=None):
    """Plot the variance of the link coordinates in function of the step in path"""

    steps, variance_x, variance_y = compute_coordinates_per_step(links_coords_finished, links_coords_unfinished)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=variance_x, mode='lines+markers', name='Variance along the horizontal axis'))
    fig.add_trace(go.Scatter(x=steps, y=variance_y, mode='lines+markers', name='Variance along the vertical axis'))

    fig.update_layout(
        title='Variance of the link coordinates (relative to the article size) in function of the step in path',
        xaxis_title='Step',
        yaxis_title='Variance',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()

    if download_path:
        fig.write_html(download_path)

def plot_graph_between(finished_paths_df, src, dst, download=True):
    """Create a graph of all pages touched by a path from src or dst"""
    DG = create_graph(finished_paths_df, src, dst)

    if download:
        Sigma.write_html(
            DG,
            f'./graph_{src}_{dst}.html',
            node_metrics=['louvain'],
            fullscreen=True,
            node_color='tag',
            node_size_range=(3, 20),
            max_categorical_colors=30,
            default_edge_type='curve',
            node_border_color_from='node',
            default_node_label_size=14,
            node_size=DG.degree)

    return Sigma(DG, 
      node_color="tag",
      node_label_size=DG.degree,
      node_size=DG.degree,
      edge_size="edge_size" 
     )

def plot_crowd_players_comparison(crowd_res, download=True):
    """Box plot of average length of individual paths and length achieved by crowd"""
    fig = go.Figure()
    fig.add_trace(go.Box(y=crowd_res['players_score'], name='Real Data'))
    fig.add_trace(go.Box(y=crowd_res['crowd_score'], name='Crowd Data'))
    fig.update_layout(
        title="Comparison of Real Path Lengths and Crowd Path Lengths",
        yaxis_title="Averages",
        xaxis_title="",
        legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
    )

    fig.show()

    if download:
        fig.write_html("box_plot_crowd_vs_players.html")

def plot_crowd_players_density(crowd_res, download=True):
    """Plot density of average lengths of individual players along with density of lengths achieved by crowd"""
    fig = ff.create_distplot(
        [crowd_res['players_score'], crowd_res['crowd_score']], 
        group_labels=['Real Data', 'Crowd Data'], 
        bin_size=1,
        colors=["rgba(0, 0, 255, 0.3)", "rgba(255, 0, 0, 0.3)"]
    )
    fig.update_layout(
        title="Distribution of averages",
        yaxis_title="Density",
        xaxis_title="Path Length",
        height=800,
        legend=dict(
                orientation="h",
                yanchor="top",
                y=0.2,
                xanchor="center",
                x=0.5
            )
    )
    fig.data = [trace for trace in fig.data if trace.yaxis != 'y2']

    fig.show()

    if download:
        fig.write_html("density_crowd_vs_players.html")

def plot_path_length_distribution(crowd_res, finished_paths_df, download=True):
    """Plot length distribution for the paths where the crowd computed a longer path than individual players"""
    failed_games, paths_crowd_fail, players_res = find_defeated_crowd(crowd_res, finished_paths_df)
    for i, _ in enumerate(failed_games):
        fig = px.bar(players_res[i], x='len', y='count')
        fig.update_layout(title={
        'text': f"Length distribution for path from \"{failed_games.iloc[i]['src']}\" to \"{failed_games.iloc[i]['dst']}\"",  # Setting the title text
        'x': 0.5,                      # Centering the title
        'xanchor': 'center',           # Anchoring the title to the center
        'yanchor': 'top'               # Anchoring the title to the top
        },
        title_font=dict(size=20, color='black'),
        legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5
            ))  # Customizing font size and color
        fig.show()

        if download:
            fig.write_html("path_length_distribution_failed_games.html")


def describe_path_length_distribution(df):
    popular_pairs = df.groupby(["source", "target"]).size().reset_index(name="count")
    popular_pairs = popular_pairs.sort_values("count", ascending=False).head(50)
    
    popular_pairs_set = set(zip(popular_pairs["source"], popular_pairs["target"]))
    filtered_df = df[df.apply(lambda row: (row["source"], row["target"]) in popular_pairs_set, axis=1)]
    filtered_df = filtered_df.drop_duplicates(subset=["hashedIpAddress", "timestamp"])

    length_size = filtered_df.groupby("path_length").size().reset_index(name="count").sort_values("path_length")
    length_size.plot(x="path_length", y="count", kind="bar", figsize=(4, 2))

    length_size["count"] /= length_size["count"].sum()
    length_size["cumulative"] = length_size["count"].cumsum()
    length_size.plot(x="path_length", y="cumulative", kind="line", figsize=(4, 2))

    print(length_size[:10])