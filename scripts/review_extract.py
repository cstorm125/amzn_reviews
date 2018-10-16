import pandas as pd
import numpy as np
import gzip
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime
import glob

#const
selected = ['reviewer_nb','asin','overall','unixReviewTime','helpful_yes','helpful_no','reviewText_len',
           'reviewText_char','summary_len','summary_char','dt',
           'reviewText_compound', 'reviewText_neg', 'reviewText_neu',
           'reviewText_pos', 'summary_compound', 'summary_neg', 'summary_neu',
           'summary_pos', 'lev1','title_len', 'title_char', 'desc_len','desc_char', 'price',
           'salesRank','brand']

#path
PATH = '/Users/charin.polpanumas/cpro/reviews/reviews/'
RAW_PATH = PATH + 'raw/'
EXT_PATH = PATH + 'extract/'

#category names
review_files = glob.glob(f'{RAW_PATH}reviews/*')
cat_names = []
for fname in review_files:
    cat_names.append(fname.split('.')[0][46:])
cat_names = cat_names[14:]
    
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

def extract_rank(x):
    if (type(x)==dict):
        if (len(list(x.keys()))>0):
            return x[list(x.keys())[0]]
        else:
            return float('nan')
    else:
        return float('nan')

def word_len(x):
    if (type(x)==str):
        return len(nltk.word_tokenize(x))
    else:
        return float('nan')

def char_len(x):
    if (type(x)==str):
        return len(x)
    else:
        return float('nan')
    
def get_sentiment(x):
    sid = SentimentIntensityAnalyzer()
    if (type(x)==str):
        score_dict = sid.polarity_scores(x)
        return score_dict['compound'],score_dict['neg'],score_dict['neu'],score_dict['pos']
    else:
        return float('nan'),float('nan'),float('nan'),float('nan')

def unix_to_dt(x):
    return(datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

def load_reviews(cat_name,do_save=True):
    reviews = getDF(RAW_PATH+'reviews/reviews_'+cat_name+'.json.gz')

    #reviwer_nb
    reviewer_count = Counter(reviews['reviewerID'])
    reviewer_df = pd.DataFrame.from_dict(reviewer_count, orient='index').reset_index().reset_index()
    reviewer_df.columns = ['reviewer_nb','reviewerID','nb_review']
    reviewer_df.drop(['nb_review'],axis=1,inplace=True)
    reviews = pd.merge(reviews,reviewer_df)

    #helpful
    reviews['helpful_yes'] = reviews.helpful.map(lambda x: x[0])
    reviews['helpful_no'] = reviews.helpful.map(lambda x: x[1])

    #len
    reviews['reviewText_len'] = reviews.reviewText.map(word_len)
    reviews['reviewText_char'] = reviews.reviewText.map(char_len)
    reviews['summary_len'] = reviews.summary.map(word_len)
    reviews['summary_char'] = reviews.summary.map(char_len)

    #datetime
    reviews['dt'] = reviews.unixReviewTime.map(unix_to_dt)

    #sentiment
    reviews['reviewText_tuple']= reviews.reviewText.map(get_sentiment)
    reviews['summary_tuple']= reviews.summary.map(get_sentiment)

    #extract tuple
    reviews['reviewText_compound'] = reviews['reviewText_tuple'].map(lambda x: x[0])
    reviews['reviewText_neg'] = reviews['reviewText_tuple'].map(lambda x: x[1])
    reviews['reviewText_neu'] = reviews['reviewText_tuple'].map(lambda x: x[2])
    reviews['reviewText_pos'] = reviews['reviewText_tuple'].map(lambda x: x[3])

    reviews['summary_compound'] = reviews['summary_tuple'].map(lambda x: x[0])
    reviews['summary_neg'] = reviews['summary_tuple'].map(lambda x: x[1])
    reviews['summary_neu'] = reviews['summary_tuple'].map(lambda x: x[2])
    reviews['summary_pos'] = reviews['summary_tuple'].map(lambda x: x[3])

    #select columns
    selected_columns = ['reviewerID','reviewer_nb', 'asin', 'reviewerName', 'reviewText',
                        'overall', 'summary', 'unixReviewTime', 'helpful_yes',
                        'helpful_no', 'reviewText_len', 'reviewText_char', 'summary_len',
                        'summary_char', 'dt', 'reviewText_compound', 'reviewText_neg', 'reviewText_neu',
                        'reviewText_pos', 'summary_compound', 'summary_neg', 'summary_neu','summary_pos']
    reviews = reviews[selected_columns]
    if do_save: reviews.to_csv(EXT_PATH + 'reviews_'+cat_name+'.csv',index=False)
    return(reviews)

def load_meta(cat_name,do_save=True):
    meta = getDF(RAW_PATH + 'meta/meta_' + cat_name + '.json.gz')
    meta.salesRank =  meta.salesRank.map(extract_rank)
    
    meta['title_len'] = meta.title.map(word_len)
    meta['title_char'] = meta.title.map(char_len)
    meta['desc_len'] = meta.description.map(word_len)
    meta['desc_char'] = meta.description.map(char_len)
    
    meta['lev1'] = meta['categories'].map(lambda x: x[0][0])
    
    selected_columns = ['asin','lev1','title','title_len','title_char','price','salesRank',
                        'categories','brand','description','desc_len','desc_char']
    meta = meta[selected_columns]
    if do_save: meta.to_csv(EXT_PATH+'meta_'+cat_name+'.csv',index=False)
    return(meta)

#run
cat_names = ['Office_Products','Books','Electronics','Musical_Instrument',
       'Baby','Automotive','Digital_Music','Grocery_and_Gourmet_Food',
       'Beauty','Cell_Phones_and_Accessories', 'Pet_Supplies','Movies_and_TV',
       'CDs_and_Vinyl','Patio_Lawn_and_Garden','Video_Games','Home_and_Kitchen',
       'Kindle_Store','Tools_and_Home_Improvement','Health_and_Personal_Care',
       'Toys_and_Games','Sports_and_Outdoors','Clothing_Shoes_and_Jewelry']

for cat_name in cat_names:
    reviews = load_reviews(cat_name)
    print(f'reviews loaded for {cat_name}')
    meta = load_meta(cat_name)
    print(f'meta loaded for {cat_name}')
    combined = pd.merge(reviews,meta,how='left',on=['asin'])
    print(f'combined merged for {cat_name}')
    to_save = combined[selected]
    to_save.to_csv(EXT_PATH+'combined_'+cat_name+'.csv',index=False)
    print(f'{cat_name} done!')