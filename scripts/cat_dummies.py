import pandas as pd
import numpy as np
import gzip
import nltk
from collections import Counter
from datetime import datetime

#path
PATH = '../data/amzn/processed/'
RAW_PATH  = '../data/amzn/raw/'
PROCESSED_PATH  = '../data/amzn/processed/'

#utils
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

cat_names = ['Office_Products','Books','Electronics','Musical_Instrument',
       'Baby','Automotive','Digital_Music','Grocery_and_Gourmet_Food',
       'Beauty','Cell_Phones_and_Accessories', 'Pet_Supplies','Movies_and_TV',
       'CDs_and_Vinyl','Patio_Lawn_and_Garden','Video_Games','Home_and_Kitchen',
       'Kindle_Store','Tools_and_Home_Improvement','Health_and_Personal_Care',
       'Toys_and_Games','Sports_and_Outdoors','Clothing_Shoes_and_Jewelry']

for cat_name in cat_names:
    print(f'Loading meta for {cat_name}')
    meta = getDF(f'{RAW_PATH}meta_{cat_name}.json.gz')

    print(f'Processing meta for {cat_name}')
    meta_ = meta[['asin','categories']].copy()
    meta_['categories'] = meta_.categories.apply(lambda x: x[0])
    dum = pd.get_dummies(meta_.categories.apply(pd.Series).stack()).sum(level=0)
    meta_dum = pd.concat([meta_,dum],1)

    print(f'Loading combined for {cat_name}')
    combined = pd.read_csv(f'{PATH}combined_{cat_name}.csv')
    print(f'Merging combined and meta for {cat_name}')
    combined_cat = pd.merge(combined,meta_dum, on = 'asin')
    print(f'Saving {cat_name}')
    combined_cat.to_csv(f'{PROCESSED_PATH}combined_{cat_name}.csv', index=False)