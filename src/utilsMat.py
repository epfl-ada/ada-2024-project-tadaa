import numpy as np
import pandas as pd


def contains_category(x, category):
    if x is None : 
        return False
    for elt in x :
        if elt is None :
            continue
        if category in elt :
            return True
    return False

def get_main_category(article, categories):
    try:
        return categories[categories["article"] == article]["category"].values[0]
    except:
        return None


