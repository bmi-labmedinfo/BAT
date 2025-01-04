"""
Transformer-based functions.

Functions:

     _translation(str, MarianMTModel, MarianTokenizer) -> str
     back_translation(str, MarianMTModel, MarianTokenizer, MarianMTModel, MarianTokenizer) -> str
     _get_masked_elements(str, str, int) -> List[str]
     masked_augmentation(str, int, str, AutoTokenizer) -> List[str]
     _prompt_wrapper(str, str, int, str) -> str
     _postprocess(str) -> List[str]
     llm_generation(str, int, str, AutoTokenizer, str, callable, str, str, bool) -> List[str]
"""
from .init import *
from .openaicaller import *


# ================================================
# Back-translation functions
# ================================================
def _translation(src_text: str, model, tokenizer):
    """
    Translate the given text.
    :param src_text: source text
    :param model: mt model
    :param tokenizer: mt tokenizer
    :return: str: translated text
    """
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def back_translation(src_text: str, model1, tokenizer1, model2, tokenizer2):
    """
    Translate a (bunch of) phrase in an intermediate language and back to the source language.
    :param src_text: phrase or bunch of phrases
    :param model1: mt model to translate from src to trg lang
    :param tokenizer1: mt tokenizer of mt model1
    :param model2: mt model to translate from trg to src lang
    :param tokenizer2: mt tokenizer of mt model2
    :return: str: back-translated phrase
    """
    src_data_list = get_phrases(text=src_text)
    back_translated = []
    for src_data in tqdm(src_data_list, desc="Processing of sub-phrases"):
        translated_text = _translation(src_text=src_data, model=model1, tokenizer=tokenizer1)
        back_translated.append(_translation(src_text=translated_text, model=model2, tokenizer=tokenizer2))
    cleaned_text = [x.strip()[:-1] if x.strip()[-1] == "." else x.strip() for x in back_translated]
    new_text = ". ".join(cleaned_text)
    new_text += "."
    return new_text


# ================================================
# Masked language modelling functions
# ================================================
def _get_masked_elements(masked_sentence: str, ml_model: str, top_k: int = 3):
    """
    Provide a list of in-context words for a single-MASK word.
    :param masked_sentence: single-MASK word sentence
    :param ml_model: ml model
    :param top_k: top k
    :return: List[str]: in-context list of words
    """
    elements = []
    fill_mask = pipeline(task='fill-mask', model=ml_model, top_k=top_k)
    results = fill_mask(masked_sentence)
    for result in results:
        if result['score'] > 0.5:
            elements.append(result["token_str"])
    if elements:
        return elements  # l ist of words


def masked_augmentation(src_text: str, n_new_data: int, ml_model_name: str, ml_tokenizer):
    """
    Mask some single-token words of a (bunch of) phrase and replacing them with in-context words.
    :param src_text: (bunch of) phrase to augment
    :param n_new_data: max number of new synthetic data
    :param ml_model_name: ml model (BERT-based) to mask some words in the text
    :param ml_tokenizer: ml tokenizer
    :return: List[str]: augmented phrases
    """
    src_data_list = get_phrases(text=src_text)
    masked_sentences_list = []
    for src_data in tqdm(src_data_list, desc="Processing of sub-phrases"):
        phrase_list = []
        original_text = src_data
        single_mask_sentences = []  # [single-MASK word sentence, masked_word, word_index]
        new_masked_elements = []
        masked_idxs = []
        list_words = original_text.split()
        for i in np.arange(len(list_words)):
            temp_list = list_words.copy()
            if len(ml_tokenizer.tokenize((list_words[i]), truncation=True)) == 1 and not (has_numbers(list_words[i])):
                temp_list[i] = ml_tokenizer.mask_token
                single_mask_sentences.append([" ".join(temp_list), list_words[i], i])

        for masked_sentence in single_mask_sentences:
            new_words = _get_masked_elements(masked_sentence[0], ml_model_name, top_k=3)  # list of words
            if new_words:
                word_index = masked_sentence[2]
                masked_idxs.append(word_index)
                new_masked_elements.append(new_words)

        if all(not sublist for sublist in new_masked_elements):
            phrase_list.append(original_text)

        else:
            combinations = generate_combinations(new_masked_elements)
            if len(combinations) > n_new_data:
                combinations = random.sample(combinations, n_new_data)
            for comb in np.arange(len(combinations)):
                temp_comb_list = list_words.copy()
                for idx in np.arange(len(masked_idxs)):
                    me = list(combinations)[comb][idx]
                    temp_comb_list[masked_idxs[idx]] = me
                mlm_str = " ".join(temp_comb_list)
                phrase_list.append(mlm_str)

        masked_sentences_list.append(phrase_list)

    new_text = get_new_sentences(new_sentences=masked_sentences_list, n_new_data=n_new_data)
    return new_text


# ================================================
# LLM rephrasing functions
# ================================================
def _prompt_wrapper(context: str, medical_field: str, n_new_data: int, src_lang: str):
    """
    Compose the prompt to be given to the LLM.
    :param context: source text
    :param medical_field: medical field context (optional)
    :param n_new_data: max number of new synthetic data
    :param src_lang: source language (italian, english, french, spanish)
    :return: str: prompt to provide to the LLM
    """
    src_text = context.strip()
    if src_text[-1] not in [",", ".", "?", "!", ";"]:
        src_text += "."
    if medical_field not in ["", " ", None]:
        prompt = f"Rewrite the following sentence regarding {medical_field.strip()} in {n_new_data} different ways " \
                 f"in {src_lang}"
    else:
        prompt = f"Rewrite the following sentence regarding the medical field in {n_new_data} different ways " \
                 f"in {src_lang}"
    input_text = f"{prompt}: \"{src_text}\"\n"
    return input_text


def _postprocess(input_text: str):
    """
    Split the LLM generated answer in single rephrasing.
    :param input_text: LLM generated answer
    :return: List[str]: list of single rephrasing
    """
    cleaned_text = input_text.replace("\"", " ")
    cleaned_text = cleaned_text.strip()
    # if cleaned_text[-1] not in [",", ".", "?", "!", ";"]:
    #    cleaned_text = cleaned_text+"."

    list_item_pattern = re.compile(
        r"[^\d*\-•]*^\s*(\d+\.*\s+|\d+\)|\d+-|[a-zA-Z]\.\s+|[a-zA-Z]\)|[a-zA-Z]\.*|[-•*]\s+)(.+?)$",
        re.VERBOSE | re.MULTILINE)
    list_cleaned_text = [match.group(2).strip() for match in list_item_pattern.finditer(cleaned_text)]

    # cleaned_text = re.sub(r'^\d+\.\s*', '', cleaned_text)
    # list_cleaned_text = list(re.split(r'\n\d+\.\s*', cleaned_text))
    list_cleaned_text = [x.strip() for x in list_cleaned_text]
    return list_cleaned_text


def llm_generation(src_text: str, n_new_data: int, src_lang: str, tokenizer,
                   api_key: str = None, caller_func: callable = None, strategy: str = "all",
                   medical_field: Optional[str] = None, verbose: bool = False):
    """
    Rephrase a (bunch of) phrase while preserving the original meaning.
    :param src_text: (bunch of) phrase to augment
    :param n_new_data: max number of new synthetic data
    :param src_lang: source language code (it,en,fr,es)
    :param tokenizer: base tokenizer
    :param api_key: llm api key (mandatory)
    :param caller_func: custom function to invoke a LLM call (default GPT-4o-mini)
    :param strategy: how to interpret the source text (
                    "all" -> as a single phrase, "sentence" -> divided by individual sentences)
    :param medical_field: medical field context (optional)
    :param verbose: print intermediate elements
    :return: List[str]: augmented phrases
    """
    if caller_func is None:
        caller_func = OpenAICaller(api_key=api_key, verbose=verbose).make_call

    language = lookup_language(lang_code=src_lang)
    if strategy == "sentence":
        src_data_list = get_phrases(text=src_text)
        if len(src_data_list) >= n_new_data / 2:
            k = 2
        elif len(src_data_list) == 1:
            k = n_new_data
        elif len(src_data_list) < n_new_data / 2:
            k = int(n_new_data / 2 - 2)
        else:
            k = n_new_data
        clean_phrases = []

        for src_data in tqdm(src_data_list, desc="Processing of sub-phrases"):
            prompt = _prompt_wrapper(context=src_data, medical_field=medical_field, n_new_data=k, src_lang=language)
            max_answer_len_tok = len(tokenizer.tokenize(src_data)) * k + 100
            response = caller_func(prompt=prompt, max_answer_length=max_answer_len_tok)
            if verbose:
                print(response)
            clean_phrases.append(_postprocess(response))

        new_text = get_new_sentences(new_sentences=clean_phrases, n_new_data=n_new_data)
        return new_text

    elif strategy == "all":
        prompt = _prompt_wrapper(context=src_text, medical_field=medical_field, n_new_data=n_new_data,
                                 src_lang=language)
        response = caller_func(prompt=prompt, max_answer_length=2000)
        if verbose:
            print(response)
        new_text = _postprocess(response)
        return new_text

