"""
Knowledge-based functions.

Functions:

     _clean_vocab(List[str]) -> List[str]
     synonyms_dict_definition(str, AutoTokenizer, str, str, bool) -> dict
     med_synonym_replacement(str, int, dict) -> List[str]
     wn_language(str) -> str
     synonym_replacement(str, str, int, dict) ->
"""
from src.bioTextAugPackage.search_terms import *


def _clean_vocab(vocabulary: List[str]):
    """
    Create a vocabulary of unique words contained in the source data, excluding numbers and inflections.
    :param vocabulary: words in the source data
    :return: List[str]: clean vocabulary
    """
    base_index = []
    base_w = []
    vocab = list(vocabulary)
    for w in np.arange(len(vocab)):
        word = vocab[w]
        if (not has_numbers(word)) and (word[:-1] not in base_w) and len(word) > 3:
            base_w.append(word[:-1])
            base_index.append(w)
    new_vocab = [vocab[i] for i in base_index]
    return new_vocab


def synonyms_dict_definition(src_data: str, src_tokenizer, src_lang: str, api_key: str, verbose: bool = False):
    """
    Creation of a synonym dictionary for the medical terms found in the source data.
    :param src_data: source data
    :param src_tokenizer: base tokenizer
    :param src_lang: source language code (it,en,fr,es)
    :param api_key: UMLS api key
    :param verbose: print intermediate elements
    :return: dict: synonym dictionary
    """
    tokenized_sentence = tokenize_sentence(src_data, src_tokenizer)
    tokenized_sentence = remove_punctuation(tokenized_sentence)
    detokenized_sentence = detokenize_sentence(tokenized_sentence, src_tokenizer)
    vocab_terms = list(set(_clean_vocab(detokenized_sentence.split())))

    synonyms_dict = {}
    sabs = lookup_sabs(src_lang_code=src_lang)

    for v in tqdm(np.arange(len(vocab_terms)), desc="Synonyms dict definition"):
        concepts = []

        # searching the code from the given term
        codes = get_codes_for_term(apikey=api_key, text=vocab_terms[v], sabs=sabs)
        if codes:
            if verbose: print(codes)
            # searching CUI from code
            cui_codes = get_cui_for_code(apikey=api_key, codes=codes, sabs=sabs)
            if cui_codes:
                if verbose: print(cui_codes)
                # searching synonyms (in sabs) from CUI
                concepts = retrieve_names_from_cui(apikey=api_key, codes=cui_codes, sabs=sabs)

        if concepts:
            if verbose: print(concepts)
            synonyms = list(set(concepts))
            synonyms_dict[vocab_terms[v]] = synonyms

    synonyms_dict = {key.lower(): [v.lower() for v in value] if isinstance(value, list) and all(
        isinstance(v, str) for v in value) else value for key, value in synonyms_dict.items()}
    # Deleting keys with no synonyms
    del_k = [k for k, v in synonyms_dict.items() if not v]
    for k in del_k:
        del synonyms_dict[k]
    # Deleting the term itself if present
    for k in synonyms_dict:
        if k in synonyms_dict[k]:
            synonyms_dict[k].remove(k)
    return synonyms_dict


# ================================================
# Med-synonym replacement function
# ================================================
def med_synonym_replacement(text: str, n_new_data: int, synonyms_dict: dict = {}):
    """
    Replace the medical terms identified in the source data with a UMLS synonym.
    :param text: source data (group of sentences)
    :param n_new_data: max number of new data
    :param synonyms_dict: source data medical terms synonym dictionary
    :return: list of new data
    """
    del_k = [k for k, v in synonyms_dict.items() if not v]
    for k in del_k:
        del synonyms_dict[k]
    src_data_list = get_phrases(text=text)
    med_synonyms_sentences = []

    for src_data in tqdm(src_data_list, desc="Processing of sub-phrases"):
        phrase_list = []
        syn = []
        index_in_context = []
        ld = src_data.split()
        for n in np.arange(len(ld)):
            if ld[n].lower() in list(synonyms_dict.keys()):
                syn.append(synonyms_dict[ld[n].lower()])
                index_in_context.append(n)
        if len(syn) == 0:
            phrase_list.append(src_data)
        else:
            if len(index_in_context) == 1:
                combinations = list(syn[0])
            else:
                combinations = generate_combinations(syn)
            if len(combinations) > n_new_data:
                combinations = random.sample(combinations, n_new_data)

            for comb in np.arange(len(combinations)):
                for h in np.arange(len(index_in_context)):
                    if len(index_in_context) == 1:
                        ld[index_in_context[h]] = combinations[comb]
                    else:
                        ld[index_in_context[h]] = combinations[comb][h]
                phrase_list.append(" ".join(ld))
        med_synonyms_sentences.append(phrase_list)

    new_text = get_new_sentences(new_sentences=med_synonyms_sentences, n_new_data=n_new_data)
    return new_text


# ================================================
# Synonym replacement function
# ================================================
def wn_language(language: str):
    """
    Find wordnet code for the given language.
    :param language: language (italian,english,french,spanish)
    :return: str: language code (it,en,fr,es)
    """
    lang_dict = {"italian": "ita", "english": "eng", "french": "fra", "spanish": "spa"}
    # "japanese": "jpn", "arabic": "arb"}
    return lang_dict[language]


def synonym_replacement(text: str, src_lang: str, n_new_data: int, med_synonyms_dict: Optional[dict] = None):
    """
    Replace the terms identified in the source data with a wordnet synonym, excluding stopwords.
    If the med_synonym_dict is provided, the words it contains are omitted.
    :param text: source data (group of sentences)
    :param src_lang: source language code (it,en,fr,es)
    :param n_new_data: max number of new data
    :param med_synonyms_dict: source data medical terms synonym dictionary (opional)
    :return: list of new data
    """
    if med_synonyms_dict is None:
        med_synonyms_dict = {}
    src_data_list = get_phrases(text=text)
    synonyms_sentences = []

    for src_data in tqdm(src_data_list, desc="Processing of sub-phrases"):
        ld = src_data.split()
        phrase_list = []
        syn = []
        index_in_context = []

        src_lang_sw = lookup_language(lang_code=src_lang)
        src_lang_wn_syn = wn_language(language=src_lang_sw)
        for n in np.arange(len(ld)):
            if ld[n].lower() not in [med_synonyms_dict.keys()] + list(set(stopwords.words(src_lang_sw))):
                synonyms = wn.synonyms(ld[n], lang=src_lang_wn_syn)
                if not all(not sublist for sublist in synonyms):
                    single_list_synonyms = [el for sublist in synonyms for el in sublist]
                    if len(single_list_synonyms) > n_new_data:
                        synonym = random.sample(single_list_synonyms, n_new_data)
                    else:
                        synonym = single_list_synonyms
                    index_in_context.append(n)
                    syn.append(synonym)

        if len(syn) == 0:
            phrase_list.append(src_data)
        else:
            combinations = generate_combinations(syn)
            if len(combinations) > n_new_data:
                combinations = random.sample(combinations, n_new_data)

            for comb in np.arange(len(combinations)):
                for h in np.arange(len(index_in_context)):
                    if list(combinations)[comb][h] != "":
                        ld[index_in_context[h]] = list(combinations)[comb][h]

                phrase_list.append(" ".join(ld))
        synonyms_sentences.append(phrase_list)

    new_text = get_new_sentences(new_sentences=synonyms_sentences, n_new_data=n_new_data)
    return new_text
