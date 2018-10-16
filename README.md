# Amazon Review Codebook

This notebook details our data processing from [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/links.html) (He and McAuley, 2016 and McAuley et al, 2015). We process the dataset for category `Musical Instruments`. We replicate the method to all categories in `scripts/review_extract.py` and `scripts/cat_dummies.py`.

```
This dataset contains product reviews and metadata from Amazon, 
including 142.8 million reviews spanning May 1996 - July 2014.
This dataset includes reviews (ratings, text, helpfulness votes), 
product metadata (descriptions, category information, price, brand, and image features), 
and links (also viewed/also bought graphs).
```

We used the [aggressively deduplicated data](http://snap.stanford.edu/data/amazon/productGraph/aggressive_dedup.json.gz) for product reviews and [product meta data](http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz) as raw data.
