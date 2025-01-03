"""
Functions for general purposes and environment configuration.

Classes:

    Config
    StringColors

Functions:

    lookup_language(str, str) -> str
    has_numbers(str) -> bool
    generate_combinations(List[str]) -> List[str]
    tokenize_sentence(str, AutoTokenizer) -> List[str]
    detokenize_sentence(str, AutoTokenizer) -> str
    remove_punctuation(str) -> List[str]
    concat_short_sentences(List[str], int) -> List[str]
    get_phrases(str) -> List[str]
    get_new_sentences(List[str], int) -> List[str]
"""

from transformers import pipeline, AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import random
import math
import torch
from typing import Optional, List
from collections import Counter
import difflib
import warnings

warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

class Config:
    """
        A class used to set config and customizable params.
        ...
        Attributes
        ----------
        umls_api_key : str
            UMLS personal API key
        llm_api_key : str
            LLM personal API key
        llm_caller: callable
            Function to invoke the LLM
        se_model_name : str
            Default multilingual sentence embedding model name
        base_tokenizer : AutoTokenizer
            Default multilingual base tokenizer
        bert_model_name : str
            Default multilingual bert-based model name
        bert_tokenizer : AutoTokenizer
            Default multilingual bert-based tokenizer

        Methods
        -------
        set_custom_llm_caller(func):
            Set a custom function to invoke an LLM of your choice.
        add_sentence_embedding_model(se_model_name):
            Set a custom sentence embedding model for sentence similarity.
        add_base_tokenizer(base_tokenizer_name):
            Set a custom base tokenizer.
        add_bert_model(bert_model_name):
            Set a custom bert-based model to compute TB-masked_lm technique.
        set_lang_based_models(language):
            Configure custom se_model_name, base_tokenizer, and bert_model_name based on the specified language.
    """

    def __init__(self, umls_api_key: str = None, llm_api_key: str = None):
        """
        Set config params.
        :param umls_api_key: UMLS personal API key
        :param llm_api_key: LLM personal API key
        """
        self.umls_api_key = umls_api_key
        self.llm_api_key = llm_api_key

        self.llm_caller = None

        # Default models are multilingual
        self.se_model_name = "Alibaba-NLP/gte-multilingual-base"
        self.base_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased",
                                                            clean_up_tokenization_spaces=True, model_max_length=512)
        self.bert_model_name = "google-bert/bert-base-multilingual-cased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, clean_up_tokenization_spaces=True)

    def set_custom_llm_caller(self, func: callable):
        """
        Set up a custom function to invoke an LLM of your choice.
        :param func: a function that accepts as input
                    1) prompt
                    2) max_answer_length
                    and returns the response as output.
        """
        if not callable(func):
            print(f"{StringColors.WARNING}> You must insert a callable function!")
            sys.exit(1)
        self.llm_caller = func

    def add_sentence_embedding_model(self, se_model_name: str):
        """
        Set a custom sentence embedding model for sentence similarity.
        :param se_model_name: sentence embedding model name
        """
        try:
            AutoTokenizer.from_pretrained(se_model_name, clean_up_tokenization_spaces=True)
        except:
            print(f"{StringColors.WARNING}> {se_model_name} inserted is not valid!")
            sys.exit(1)
        self.se_model_name = se_model_name

    def add_base_tokenizer(self, base_tokenizer_name: str):
        """
        Set a custom base tokenizer.
        :param base_tokenizer_name: tokenizer name
        """
        try:
            AutoTokenizer.from_pretrained(base_tokenizer_name, clean_up_tokenization_spaces=True)
        except:
            print(f"{StringColors.WARNING}> {base_tokenizer_name} inserted is not valid!")
            sys.exit(1)
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, clean_up_tokenization_spaces=True)

    def add_bert_model(self, bert_model_name: str):
        """
        Set a custom bert-based model to compute TB-masked_lm technique.
        :param bert_model_name: bert-based model name
        """
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name, clean_up_tokenization_spaces=True)
        except:
            print(f"{StringColors.WARNING}> {bert_model_name} inserted is not valid!")
            sys.exit(1)
        self.bert_model_name = bert_model_name

    def set_lang_based_models(self, language: str):
        """
        Configure custom se_model_name, base_tokenizer, and bert_model_name based on the specified language.
        :param language: source text language (italian, english, spanish, french)
        """
        se_models = {
            "italian": "efederici/sentence-it5-base",
            "english": "sentence-transformers/all-MiniLM-L6-v2",
            "french": "dangvantuan/sentence-camembert-base",
            "spanish": "hiiamsid/sentence_similarity_spanish_es"
        }
        base_models = {
            "italian": "it5/it5-base-question-answering",
            "english": "google-t5/t5-base",
            "french": "google-bert/bert-base-multilingual-cased",
            "spanish": "google-bert/bert-base-multilingual-cased"
        }
        bert_models = {
            "italian": "IVN-RIN/bioBIT",
            "english": "dmis-lab/biobert-v1.1",
            "french": "almanach/camembert-bio-base",
            "spanish": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        }
        self.se_model_name = se_models[language]
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_models[language],
                                                            clean_up_tokenization_spaces=True, model_max_length=512)
        self.bert_model_name = bert_models[language]
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, clean_up_tokenization_spaces=True)


class StringColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def lookup_language(language: str = None, lang_code: str = None):
    """
    Conversion from language to lang_code and vice versa.
    :param language: language
    :param lang_code: language encoding
    :return: str: lang_code or language
    """
    lang_dict = {"italian": "it", "english": "en", "french": "fr", "spanish": "es", "german": "de",
                 "japanese": "ja", "chinese": "zh", "arabic": "ar"}
    if language is not None:
        if language.lower() in lang_dict.keys():
            return lang_dict[language.lower()]
        else:
            print(f"{StringColors.WARNING}> The source language inserted is not valid! Possible languages are:"
                  f"italian, english, french, spanish")
            sys.exit(1)
    if lang_code is not None:
        for k, v in lang_dict.items():
            if v == lang_code:
                return k


def has_numbers(sentence: str):
    pattern = r'\d'  # every number
    return bool(re.search(pattern, sentence))


def generate_combinations(vectors):
    """
    Generate various combinations of the provided words.
    :param vectors: list of word lists
    :return: Tuple: tuple containing the combinations
    """
    combinations = list(product(*vectors))
    return combinations


def tokenize_sentence(sentence: str, src_tokenizer):
    return src_tokenizer.tokenize(sentence.lower())


def detokenize_sentence(sentence: str, src_tokenizer):
    return src_tokenizer.convert_tokens_to_string(sentence)


def remove_punctuation(sentence: str):
    list_words = []
    for word in sentence:
        if word not in set(string.punctuation):
            list_words.append(word)
    return list_words


def concat_short_sentences(sentences: List[str], min_n_char: int = 30):
    """
    Combine multiple sentences with a character count below the specified limit.
    :param sentences: list of sentences
    :param min_n_char: minimum number of char
    :return: List[str]: list of sentences
    """
    concatenated_sentences = []
    current_sentence = sentences[0]

    for sentence in sentences[1:]:
        if len(sentence) < min_n_char:
            current_sentence += ". " + sentence
        else:
            concatenated_sentences.append(current_sentence)
            current_sentence = sentence
    concatenated_sentences.append(current_sentence)
    return concatenated_sentences


def get_phrases(text: str):
    """
    Split the given text in phrases.
    :param text: source text
    :return: List[str]: list of phrases
    """
    src_data_list = list(filter(None, [el for el in text.split(". ")]))
    src_data_list = concat_short_sentences(src_data_list)
    if src_data_list[-1][-1] == ".":
        src_data_list[-1] = src_data_list[-1][:-1]
    return src_data_list


def get_new_sentences(new_sentences, n_new_data: int):
    """
    Combine the synthetic sentences to generate final new ones.
    :param new_sentences: new sentences generated by a technique
    :param n_new_data: number of new synthetic data
    :return: List[str]: new text
    """
    new_text = []
    combinations = generate_combinations(new_sentences)
    if len(combinations) > n_new_data:
        combinations = random.sample(combinations, n_new_data)
    for comb in np.arange(len(combinations)):
        if combinations[comb][0][-1] != ".":
            new_text.append(". ".join(combinations[comb]))
        else:
            new_text.append(" ".join(combinations[comb]))
    new_text = [x.strip() + "." if x.strip()[-1] != "." else x.strip() for x in new_text]
    return new_text
