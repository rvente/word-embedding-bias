#!/usr/bin/python3
# Load the package
import json
from smart_open import smart_open, open as s_open
import logging
from operator import ge
from scipy.stats.stats import mode
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics.WEAT import WEAT
from wefe.datasets.datasets import load_weat
from wefe.utils import run_queries, plot_queries_results

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import ParameterGrid
import pandas as pd
from pathlib import Path
from datetime import datetime
from gensim.corpora import WikiCorpus
import re


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# https://radimrehurek.com/gensim/scripts/word2vec_standalone.html

RESULTS_DIR = Path('./results/')
RUN_TIME = f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"


class WikiPartitionCorpus:
    """ 
    utility class that takes a partition of the wikipedia corpus
    """

    def __init__(self, fname, max_sentence_length=100000):
        """Iterate over sentences from a corpus, in .gz form"""
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        with smart_open(self.fname, 'rb', encoding="utf-8") as infile:
            for row in infile:
                s = " ".join(json.loads(row)['section_texts']).lower()
                s = re.sub(r'[^a-z\']+', ' ', s)  # keep alpha
                s = re.sub(r"''", ' ', s)
                yield s.split()
                # for chunk in chunks(s.split(), n=10000):
                #     yield chunk


def train_model(architecture: Word2Vec or FastText, dataset, dset_fname, window_size, min_words=1, vector_dims=20, n_workers=16, n_epochs=5):
    """
    instantiate a model and rain it
    """

    model = architecture(
        workers=n_workers, epochs=n_epochs,  window=window_size, min_count=min_words,
    )
    model.build_vocab(
        corpus_file=dset_fname
        # corpus_iterable=dataset
    )
    model.train(
        epochs=n_epochs,  window=window_size, min_count=min_words,
        vector_size=vector_dims,
        compute_loss=True,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        corpus_file=dset_fname
        # sentences=dataset,
    )
    return model


def wv_to_weat(wv: KeyedVectors, gender_query):
    return WEAT().run_query(
        gender_query,
        WordEmbeddingModel(wv),
        return_effect_size=True,
        calculate_p_value=True,
        preprocessor_args={
            # 'lowercase': True,
            'strip_accents': True
        })


def term_pair_to_query(human_axis=('male_terms', 'female_terms'), term_axis=('career', 'family')):
    word_sets = load_weat()
    human_1, human_2 = human_axis
    term_1, term_2 = term_axis
    gender_query = Query([word_sets[human_1], word_sets[human_2]],
                         [word_sets[term_1], word_sets[term_2]],
                         [human_1, human_2], [term_1, term_2])
    return gender_query


def word_similarity(wv: KeyedVectors):
    """
    perform wordsim on 
    """
    return wv.evaluate_word_pairs(datapath('wordsim353.tsv'), dummy4unknown=True)


def evaluate_analogies(wv: KeyedVectors):
    return wv.evaluate_word_analogies(datapath('questions-words.txt'), dummy4unknown=True)


if __name__ == "__main__":
    # wiki_path = api.load("wiki-english-20171001", return_path=True)
    wiki_path = "/path/to/your/processed_file.txt"
    wikipedia_english = [WikiPartitionCorpus(wiki_path)]

    datasets = [wikipedia_english]
    epochs = [1]

    queries = list(map(lambda d: term_pair_to_query(**d), [
        dict(human_axis=('male_terms', 'female_terms'),
             term_axis=('career', 'family')),
        dict(human_axis=('male_terms', 'female_terms'),
             term_axis=('science', 'arts')),
        dict(human_axis=('male_terms', 'female_terms'),
             term_axis=('math', 'arts')),
    ]))

    # define parameters to iterate through
    architectures = [Word2Vec, FastText]
    window_sizes = [1, 5, 10, 15, 20, 25, 30]
    vector_dims = [300]

    param_grid = {
        'architecture': architectures,
        'window_size': window_sizes,
        'dataset': datasets,
        'n_epochs': epochs,
        'vector_dims': vector_dims,
        'dset_fname': [wiki_path],
    }

    METRICS = ["gender_results", "analogy", 'word_sim']
    res_df = pd.DataFrame(columns=[*param_grid.keys(), *METRICS])
    for i, params in enumerate(ParameterGrid(param_grid)):

        # parallel training
        model = train_model(**params)
        df_params = {k: str(v) for k, v in params.items()}
        keyed_vecs = model.wv

        # evaluation metrics
        wordsim = word_similarity(keyed_vecs)
        analogy = evaluate_analogies(keyed_vecs)
        gender_results = [wv_to_weat(keyed_vecs, query) for query in queries]

        res_df.loc[i] = {
            **df_params,
            'gender_results': pd.DataFrame(gender_results),
            'analogy': analogy,
            'word_sim': wordsim
        }

        # overwrite w each partial result
        res_df.to_pickle(RESULTS_DIR/RUN_TIME)

    logging.info(res_df)
