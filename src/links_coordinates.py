import pandas as pd
import networkx as nx

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os


def compute_shortest_path(source, target, graph):
    """
    Compute the shortest path between source and target nodes in a graph.

    Args:
        source: source article name
        target: target article name
        graph: networkx graph

    Returns:
        The shortest path between source and target nodes in the graph
    """
    try:
        shortest_path = nx.shortest_path(graph, source=source, target=target)
        return shortest_path
    except nx.NetworkXNoPath:
        return None


def get_article_html_path(article_name):
    """
    Get the path of the html file of the article.

    Args:
        article_name: article name

    Returns:
        The path of the html file of the article
    """
    return os.path.abspath("data/wpcd/wp/{}/{}.htm".format(article_name[0].lower(), article_name))


def get_path_links_coordinates(browser, articles, cache, normalized=True):
    """
    Get the coordinates of the links between articles in the path.

    Args:
        browser: selenium webdriver used to simulate the browser
        articles: list of articles in the path
        cache: dictionary to store the coordinates of the links between articles
        normalized: whether to normalize the coordinates or not
    
    Returns:
        A list of coordinates of the links between articles in the path
    """
    path_links_coords = []
    for i in range(len(articles) - 1):
        cur_article = articles[i]
        next_article = articles[i+1]
        if cur_article == "<":
            continue
        if cur_article in cache and next_article in cache[cur_article]:
            path_links_coords.append(cache[cur_article][next_article])
            print("Cache hit for {} -> {}".format(cur_article, next_article))
            continue
        
        local_html_file = get_article_html_path(cur_article)

        print("Opening file: ", local_html_file)
        browser.get("file:///" + local_html_file)

        next_url = "../../wp/{}/{}.htm".format(next_article[0].lower(), next_article)

        links = browser.find_elements(By.XPATH, "//a[@href=\"{}\"]".format(next_url))
          
        links_coords = []
        for link in links:
            x = link.location["x"]
            y = link.location["y"]
            if normalized:
                page_width = browser.execute_script("return document.body.scrollWidth")
                page_height = browser.execute_script("return document.body.scrollHeight")

                x /= page_width
                y /= page_height
            
            links_coords.append((x, y))

        if cache.get(cur_article) is None:
            cache[cur_article] = {}
        cache[cur_article][next_article] = links_coords

        path_links_coords.append(links_coords)
    
    return path_links_coords

def get_path_page_size(browser, articles, cache):
    """
    Get the size of the page of the articles in the path.

    Args:
        browser: selenium webdriver used to simulate the browser
        articles: list of articles in the path
        cache: dictionary to store the size of the page of the articles
    
    Returns:
        A list of the size of the page of the articles in the path
    """
    path_page_size = []
    for article in articles:
        if article in cache:
            path_page_size.append(cache[article])
            print("Cache hit for {}".format(article))
            continue
        
        local_html_file = get_article_html_path(article)

        print("Opening file: ", local_html_file)
        browser.get("file:///" + local_html_file)
        
        page_width = browser.execute_script("return document.body.scrollWidth")
        page_height = browser.execute_script("return document.body.scrollHeight")

        cache[article] = (page_width, page_height)
        path_page_size.append((page_width, page_height))
    
    return path_page_size


if __name__ == "__main__":
    finished_paths = pd.read_csv("data/clean_data/clean_finished_paths.csv")
    unfinished_paths = pd.read_csv("data/clean_data/clean_unfinished_paths.csv")

    print(finished_paths.shape)
    print(unfinished_paths.shape)

    popular_pairs = unfinished_paths.groupby(["source", "target"]).size().reset_index(name="count")
    most_popular_pairs = popular_pairs.sort_values("count", ascending=False).head(50)

    links = pd.read_csv("data/wikispeedia_paths-and-graph/links.tsv", sep="\t", skiprows= 11, names = ["source", "target"])
    graph = nx.from_pandas_edgelist(links, source="source", target="target", create_using=nx.DiGraph)
    most_popular_pairs["shortest_path"] = most_popular_pairs.apply(lambda r: compute_shortest_path(r["source"], r["target"], graph), axis=1)

    # setup
    cache = {}
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    browser = webdriver.Chrome(options=chrome_options)

    # optimal paths
    most_popular_pairs["links_coords"] = most_popular_pairs["shortest_path"].apply(lambda p: get_path_links_coordinates(browser, p, cache))
    most_popular_pairs.to_csv("data/links_coordinates_optimal.csv", index=False)

    # unfinished paths
    popular_pairs_set = set(zip(most_popular_pairs["source"], most_popular_pairs["target"]))
    filtered_unfinished_paths = unfinished_paths[unfinished_paths.apply(lambda row: (row["source"], row["target"]) in popular_pairs_set, axis=1)]
    filtered_unfinished_paths = filtered_unfinished_paths.drop_duplicates(subset=["hashedIpAddress", "timestamp"])

    filtered_unfinished_paths["links_coords"] = filtered_unfinished_paths["path"].apply(eval).apply(lambda p: get_path_links_coordinates(browser, p, cache))
    filtered_unfinished_paths = filtered_unfinished_paths[["hashedIpAddress", "timestamp", "durationInSec", "play_duration", "clean_path", "source", "target", "links_coords"]]
    filtered_unfinished_paths.to_csv("data/links_coordinates_unfinished.csv", index=False)

    # finished paths
    popular_pairs_set = set(zip(most_popular_pairs["source"], most_popular_pairs["target"]))
    filtered_finished_paths = finished_paths[finished_paths.apply(lambda row: (row["source"], row["target"]) in popular_pairs_set, axis=1)]
    filtered_finished_paths = filtered_finished_paths.drop_duplicates(subset=["hashedIpAddress", "timestamp"])

    filtered_finished_paths["links_coords"] = filtered_finished_paths["path"].apply(eval).apply(lambda p: get_path_links_coordinates(browser, p, cache))
    filtered_finished_paths = filtered_finished_paths[["hashedIpAddress", "timestamp", "durationInSec", "clean_path", "source", "target", "links_coords"]]
    filtered_finished_paths.to_csv("data/links_coordinates_finished.csv", index=False)

    browser.quit()