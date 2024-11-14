import numpy as np
from collections import Counter
import pandas as pd
import src.utils as utils

def find_target(target, list):
    """Find the target and output its successor."""
    ret = ''
    for i in range(len(list) - 1):
        if list[i] == target:
            ret = list[i+1]
    return ret

def find_next(source, dest, finished_paths):
    """Find the next page the crowd algorithm will click on."""
    paths_src = finished_paths[finished_paths['path'].apply(lambda x: source in x and x[-1]==dest)]['path']
    next_list = [find_target(source, path) for path in paths_src]
    next_list = [elem for elem in next_list if elem != '']
    number_of_votes = len(next_list)
    if number_of_votes==0:
        return number_of_votes, '' 
    else:
        return number_of_votes, Counter(next_list).most_common(1)[0][0]
    

def crowd(src, dest, finished_paths):
    """Crowd algorithm where every players vote at each step."""
    next = src
    path = [src]
    lst_votes = []
    while(next != dest):
        nb_votes, next = find_next(next, dest, finished_paths)
        lst_votes.append(nb_votes)
        if not(next in path) and nb_votes!=0:
            path.append(next)
        else:
            break
    return path, np.array(lst_votes)

def stats_crowd(games, finished_paths):
    """Compute paths thanks to the crowd algorithm."""
    res_crowd = []
    for (src, dst) in games:
        path, votes = crowd(src, dst, finished_paths)
        if (np.all(votes > 0)):
            res_crowd.append(((src, dst), len(path)))
    return res_crowd

def stats_players(games, finished_paths):
    """Compute averages of length of paths given by individual players"""
    avg_res = []
    for (src, dst) in games:
        paths = finished_paths[finished_paths['path'].apply(lambda x: x[0]==src and x[-1]==dst)]['path']
        if len(paths)>0:
            avg_res.append(((src, dst), paths.apply(lambda p: len(p)).sum()/len(paths)))
    return avg_res

def popular_words(occ_all, count_threshold=200):
    """Output the most popular pages in any games given the occurence list of any word in the dataset."""
    return [element for element, count in occ_all.items() if count > count_threshold]

def voter_score(finished_paths, popular_words):
    """We compute the minimum number of voters in the most promising paths."""
    scores = []
    iter = 0
    for w1 in popular_words:
        for w2 in popular_words:
            print(f'iter #{iter}/{len(popular_words)**2}')
            iter += 1
            if (w1 != w2):
                res_path, votes = crowd(w1, w2, finished_paths)
                if (np.all(votes > 0) and res_path[-1]==w2):
                    scores.append(((w1, w2), np.min(votes)))
    return scores

def games_to_play(scores, score_threshold=50):
    """Clean the scores array and filter low voter_score paths."""
    return [((src, dst), count) for ((src, dst), count) in scores if count > score_threshold and src != '<' and dst != '<']


def compute_scores(finished_paths):
    """
    We compute a score for each tuple of different words.
    We take the score to be the minimum of voters for each step of the crowd algorithù.
    We think it is a fair metric as a big score attest without fail the path makes intervene a lot of players at each step.
    It would not be the case if we took just the sum of the list of votes (for example).
    We finally store in a csv file the result.
    """

    words = set()
    for p in finished_paths["path"]:
        words.update(p)
    
    occ_all = Counter()
    for p in finished_paths["path"]:
        occ_all += Counter(p)
    
    popular_words = popular_words(occ_all)

    # We compute all the scores for the most promising games
    scores = voter_score(finished_paths, popular_words)

    df = pd.DataFrame(games_to_play(scores), columns=["src_dst", "score"])
    df[["src", "dst"]] = pd.DataFrame(df["src_dst"].tolist(), index=df.index)
    df = df.drop(columns=["src_dst"])
    # Save to CSV
    df.to_csv("./data/crowd_paths.csv", index=False)

def stats_players_crowd(finished_paths):
    """
    We compute the scores achieved by the crowd vs scores achieved by individual players.
    Result is stored in a csv file.
    """
    games_to_play = pd.read_csv("./data/crowd_paths.csv")

    ans_crowd = []
    for (src, dst) in games_to_play[['src', 'dst']].values:
        ans_crowd.append(crowd(src, dst, finished_paths))
    
    ans_stats = []
    for (src, dst) in games_to_play[['src', 'dst']].values:
        paths = finished_paths[finished_paths['path'].apply(lambda x: src in x and x[-1]==dst)]['path']
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
    data.to_csv("./data/crowd_vs_players.csv.csv", index=False)
    
