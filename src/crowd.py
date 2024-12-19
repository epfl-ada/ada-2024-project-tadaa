import numpy as np
from collections import Counter
import pandas as pd
import src.utils as utils
import networkx as nx

def find_target(target, path):
    """
    Find the target article and output its successor.

    Args:
        target: target article name
        path: list of articles

    Returns:
        Next article name the path jumped to (after target).
    """
    ret = ''
    for i in range(len(path) - 1):
        if path[i] == target:
            ret = path[i+1]
    return ret

def find_next(source, dest, finished_paths_df):
    """
    Find the next page the crowd algorithm will click on.
    
    Args:
        source: first article name in the game
        dest: last article name in the game
        finished_paths_df : dataframe containing every successful paths played

    Returns:
        Next article the crowd chose to click on.
    """
    paths_src = finished_paths_df[finished_paths_df['clean_path'].apply(lambda x: source in x and x[-1]==dest)]['clean_path']
    next_list = [find_target(source, path) for path in paths_src]
    next_list = [elem for elem in next_list if elem != '']
    number_of_votes = len(next_list)
    if number_of_votes==0:
        return number_of_votes, '' 
    else:
        return number_of_votes, Counter(next_list).most_common(1)[0][0]
    

def crowd(src, dest, finished_paths_df):
    """
    Crowd algorithm where every players vote at each step.

    Args:
        source: first article name in the game
        dest: last article name in the game
        finished_paths_df : dataframe containing every successful paths played

    Returns:
        path : path computed by the crowd algorithm (list)
        votes : the number of votes at each step of the algorithm (np.array)
    """
    next = src
    path = [src]
    lst_votes = []
    while(next != dest):
        nb_votes, next = find_next(next, dest, finished_paths_df)
        lst_votes.append(nb_votes)
        if not(next in path) and nb_votes!=0:
            path.append(next)
        else:
            break
    return path, np.array(lst_votes)

def stats_crowd(games, finished_paths_df):
    """
    Compute paths thanks to the crowd algorithm.
    
    Args:
        games: list of (src, dst) tuples, each represent a game to be played
        finished_paths_df : dataframe containing every successful paths played

    Returns:
        list of tuples associating a game with the length of the path found by the crowd algorithm.
    """
    res_crowd = []
    for (src, dst) in games:
        path, votes = crowd(src, dst, finished_paths_df)
        if (np.all(votes > 0)):
            res_crowd.append(((src, dst), utils.path_length(path)))
    return res_crowd

def stats_players(games, finished_paths_df):
    """
    Compute averages of length of paths given by individual players
    
    Args:
        games: list of (src, dst) tuples, each represent a game to be played
        finished_paths_df : dataframe containing every successful paths played

    Returns:
        list of tuples associating a game with the average length of the paths achieved by individual players.
    """
    avg_res = []
    for (src, dst) in games:
        paths = finished_paths_df[finished_paths_df['clean_path'].apply(lambda x: x[0]==src and x[-1]==dst)]['clean_path']
        if len(paths)>0:
            avg_res.append(((src, dst), paths.apply(lambda p: len(p)).sum()/len(paths)))
    return avg_res

def popular_words(occ_all, count_threshold=200):
    """
    Output the most popular articles in any games given the occurence list of any word in the dataset.

    Args:
        occ_all : occurence of every article names present in every successful paths played (Counter)
        count_threshold : threshold indicating from which value we retain an article name (int)

    Returns:
        list of words that appear more than count_thresold times.
    """
    return [element for element, count in occ_all.items() if count > count_threshold]

def voter_score(finished_paths_df, popular_words):
    """
    Compute the minimum number of voters in the most promising paths.

    Args:
        finished_paths_df : dataframe containing every successful paths played
        popular_words : list of article names that appeared a significant number of times

    Returns:
        List of tuples associating a game with the correspoding voter-score achieved during crowd algorithm.
    """
    scores = []
    iter = 0
    for w1 in popular_words:
        for w2 in popular_words:
            print(f'iter #{iter}/{len(popular_words)**2}')
            iter += 1
            if (w1 != w2):
                res_path, votes = crowd(w1, w2, finished_paths_df)
                if (np.all(votes > 0) and res_path[-1]==w2):
                    scores.append(((w1, w2), np.min(votes)))
    return scores

def games_to_play(scores, score_threshold=50):
    """
    Clean the scores array and filter low voter-score paths.
    
    Args:
        scores : list of voter-scores for each game
        score_threshold : threshold indicating from which value we retain a game (int)

    Returns:
        Cleaned list of tuples associating a game with the correspoding voter-score.
    """
    return [((src, dst), count) for ((src, dst), count) in scores if count > score_threshold and src != '<' and dst != '<']


def compute_scores(finished_paths_df):
    """
    Compute a score for each tuple of different words. We take the score to be the minimum of voters
    for each step of the crowd algorithm. It is a fair metric as a big score attest the path makes intervene 
    a lot of players at each step. It would not be the case if we took just the sum of the list of votes (for example). 
    We finally store the result in a csv file.

    Args:
        finished_paths_df : dataframe containing every successful paths played

    Returns:
        None
    """

    words = set()
    for p in finished_paths_df["path"]:
        words.update(p)
    
    occ_all = Counter()
    for p in finished_paths_df["path"]:
        occ_all += Counter(p)
    
    popular_words_ = popular_words(occ_all)

    # We compute all the scores for the most promising games
    scores = voter_score(finished_paths_df, popular_words_)

    df = pd.DataFrame(games_to_play(scores), columns=["src_dst", "score"])
    df[["src", "dst"]] = pd.DataFrame(df["src_dst"].tolist(), index=df.index)
    df = df.drop(columns=["src_dst"])
    # Save to CSV
    df.to_csv("./data/crowd_paths.csv", index=False)

def stats_players_crowd(finished_paths_df):
    """
    Compute the scores achieved by the crowd & scores achieved by individual players.
    Result is stored in a csv file.

    Args:
        finished_paths_df : dataframe containing every successful paths played
    
    Returns:
        None
    """
    games_to_play = pd.read_csv("./data/crowd_paths.csv")

    ans_crowd = []
    for (src, dst) in games_to_play[['src', 'dst']].values:
        ans_crowd.append(crowd(src, dst, finished_paths_df))
    
    ans_stats = []
    for (src, dst) in games_to_play[['src', 'dst']].values:
        paths = finished_paths_df[finished_paths_df['clean_path'].apply(lambda x: src in x and x[-1]==dst)]['clean_path']
        # Find the index of the first occurrence of the target element
        paths_src = []
        for p in paths:
            start_index = p.index(src)
            paths_src.append(p[start_index:])
        if len(paths_src) == 0:
            ans_stats.append(-1)
        else:
            ans_stats.append(sum(utils.path_length(p) for p in paths_src) / len(paths_src))
    
    ans_stats = np.array(ans_stats)

    data = pd.DataFrame()
    data['src'] = [path[0] for (path, scores) in ans_crowd]
    data['dst'] = [path[-1] for (path, scores) in ans_crowd]
    data['crowd_score'] = [utils.path_length(path) for (path, scores) in ans_crowd]
    data['players_score'] = ans_stats
    # Save to CSV
    data.to_csv("./data/crowd_vs_players.csv", index=False)

def find_defeated_crowd(games_to_play, ans_stats, ans_crowd, crowd_stats):
    """
    Find all games where the crowd found a longer path than the average length of indivual paths 
    along with the list of coressponding average length of such individual paths

    Args:
        games_to_play : numpy array (n,) with all games
        ans_stats : numpy array (n,) with all averages of individual path lengths
        ans_crowd : numpy array (n,) with all paths computed by crowd
        crowd_stats : numpy array (n,) with all path lengths achieved by crowd
    
    Returns:
        crowd_fail_ids : list of games where the crowd achieved a longer path than the average length of indivual paths
        paths_crowd_fail : list of corresponding paths computed by the crowd
    """
    crowd_fail_ids = np.nonzero((ans_stats - crowd_stats) < 0)[0]
    paths_crowd = [p[0] for p in ans_crowd]
    paths_crowd_fail = [p for (i, p) in enumerate(paths_crowd) if i in crowd_fail_ids]
    return np.array(games_to_play)[crowd_fail_ids, :], paths_crowd_fail

def edges_from_path(path):
    """
    Construct a list of edges from a path that is a list of strings

    Args:
        path: numpy array (n,). An arbitrary path
    
    Returns:
        list of tuples of names following the order in which they appear in path
    """
    edges = []
    for i in range(len(path)):
        if i+1 < len(path):
            edges.append((path[i], path[i+1]))
    return edges

def create_graph(finished_paths, src, dst):
    """
    Build a graph from all paths that link src and dst

    Args:
        finished_paths: dataframe containing every successful paths played
        src: source node (string)
        dst: sink node (string)
    
    Returns:
        Directed Graph computed from paths that link src and dst
    """
    DG = nx.DiGraph()

    paths = finished_paths[finished_paths['clean_path'].apply(lambda x: src in x and x[-1]==dst)]['clean_path']
    # Find the index of the first occurrence of the target element
    paths_src = []
    for p in paths:
        start_index = p.index(src)
        paths_src.append(p[start_index:])

    for p in paths_src:
        for (u, v) in edges_from_path(p):
            if DG.has_edge(u, v):
                DG[u][v]['weight'] += 1
            else:
                DG.add_edge(u, v)
                DG[u][v]['weight'] = 1

    # Set attributes for source and sink nodes
    nx.set_node_attributes(DG, {src: "source", dst: "sink"}, name="tag")
    for node in DG.nodes:
        if "tag" not in DG.nodes[node]:
            DG.nodes[node]["tag"] = "intermediate"

    path_crowd = ['Herbivore', 'Animal', 'Mammal', 'Zebra']
    nx.set_edge_attributes(DG, {e:"red"for e in edges_from_path(path_crowd)}, name="color")

    nx.set_node_attributes(DG, {src: 0, dst: 1}, name="x")
    nx.set_node_attributes(DG, {src: 0.5, dst: 0.5}, name="y")

    # Remove lonely nodes
    nodes_to_remove = [node for node, degree in DG.degree() if degree <= 2]
    DG.remove_nodes_from(nodes_to_remove)

    return DG
