import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib.parse import unquote


import os

def load_dataframe(file_path: str, skip_rows: int, columns: list) -> pd.DataFrame:
    """Load a DataFrame from a file."""
    df = pd.read_csv(file_path, sep="\t", skiprows= skip_rows, header=None)
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
        paths_df["rating"].fillna(-1, inplace=True)
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


def compute_shortest_path(source, target, graph):
    try:
        shortest_path = nx.shortest_path(graph, source=source, target=target)
        return shortest_path
    except nx.NetworkXNoPath:
        return None


def create_graph(links_df: pd.DataFrame) -> nx.DiGraph:
    """Create a graph from the links DataFrame."""
    graph = nx.from_pandas_edgelist(links_df,
        source="source", target="target", create_using=nx.DiGraph)
    return graph
def get_article_html_path(article_name):
    return os.path.abspath("data/wpcd/wp/{}/{}.htm".format(article_name[0].lower(), article_name))


def get_path_links_coordinates(browser, articles, cache):
    path_links_coords = []
    for i in range(len(articles) - 1):
        cur_article = articles[i]
        next_article = articles[i + 1]
        if cur_article == "<":
            continue
        if cur_article in cache and next_article in cache[cur_article]:
            path_links_coords.extend(cache[cur_article][next_article])
            print("Cache hit for {} -> {}".format(cur_article, next_article))
            continue

        local_html_file = get_article_html_path(cur_article)

        print("Opening file: ", local_html_file)
        browser.get("file:///" + local_html_file)

        next_url = "../../wp/{}/{}.htm".format(next_article[0].lower(), next_article)

        links = browser.find_elements(By.XPATH, "//a[@href=\"{}\"]".format(next_url))
        links_coords = [(link.location["x"], link.location["y"]) for link in
                        links]  # if many links are found we take all their coordinates

        if cache.get(cur_article) is None:
            cache[cur_article] = {}
        cache[cur_article][next_article] = links_coords

        path_links_coords.extend(links_coords)

    return path_links_coords