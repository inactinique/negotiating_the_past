# Negotiating the past

Set up your environment.

```
# Create a new conda environment with Python 3.11
conda create -n prompting_past python=3.11

# Activate the environment
conda activate prompting_past

# Install necessary packages
pip install pandas numpy scikit-learn umap-learn hdbscan matplotlib seaborn plotly
pip install sentence-transformers torch nltk gensim spacy
pip install jupyter notebook

# If you need specialized NLP packages
python -m spacy download en_core_web_md
python -m nltk.downloader punkt stopwords wordnet
```

