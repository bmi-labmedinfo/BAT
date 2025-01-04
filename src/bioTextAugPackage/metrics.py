"""
Functions for calculating metrics.

Functions:

    compute_overlap(str, str, AutoTokenizer) -> float
    compute_similarity(str, str, str) -> float
"""

from .init import *


def compute_overlap(synthetic_data: str, src_data: str, tokenizer):
    """
    Compute token overlap between the source textual data and the synthetic one.
    :param synthetic_data: new synthetic data
    :param src_data: source data
    :param tokenizer: base tokenizer
    :return: f1_overlap
    """
    src_data_list = get_phrases(text=src_data)
    synthetic_data_list = get_phrases(text=synthetic_data)
    mean_overlap = []

    for src_text, synth_text in zip(src_data_list, synthetic_data_list):
        synth_tokens = tokenizer.tokenize(synth_text)
        src_tokens = tokenizer.tokenize(src_text)
        if synth_tokens == src_tokens:
            mean_overlap.append(1)
            continue
        common_tokens = Counter(synth_tokens) & Counter(src_tokens)
        num_same = sum(common_tokens.values())
        # if no answer, f1=1 if they agree and 0 otherwise
        if len(src_tokens) == 0 or len(synth_tokens) == 0:
            mean_overlap.append(int(src_tokens == synth_tokens))
            continue
        # if there are no common tokens then f1 = 0
        if num_same == 0:
            mean_overlap.append(0)
            continue
        precision = round(1.0 * num_same/len(synth_tokens), 2)
        recall = round(1.0 * num_same/len(src_tokens), 2)
        f1_overlap = round((2 * precision * recall) / (precision + recall), 2)
        mean_overlap.append(f1_overlap)
    return round(np.mean(mean_overlap), 2)


def compute_similarity(synthetic_data: str, src_data: str, se_model_name: str):
    """
    Compute the similarity score between the source textual data and the synthetic one.
    :param synthetic_data: new synthetic data
    :param src_data: source data
    :param se_model_name: sentence embedding model name
    :return: similarity_score
    """
    src_data_list = get_phrases(text=src_data)
    synthetic_data_list = get_phrases(text=synthetic_data)
    if se_model_name.startswith("Alibaba-NLP"):
        se_model = AutoModel.from_pretrained(se_model_name, trust_remote_code=True)
    else:
        se_model = AutoModel.from_pretrained(se_model_name)
    se_tokenizer = AutoTokenizer.from_pretrained(se_model_name, clean_up_tokenization_spaces=True)
    mean_similarity = []

    for src_text, synth_text in zip(src_data_list, synthetic_data_list):

        encoded_src = se_tokenizer(src_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        if type(synth_text) == str:
            encoded_synth = se_tokenizer(synth_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                # Compute token embeddings
                model_config = AutoConfig.from_pretrained(se_model_name) if not se_model_name.startswith("Alibaba-NLP")\
                    else AutoConfig.from_pretrained(se_model_name, trust_remote_code=True)
                if hasattr(model_config, "is_encoder_decoder") and model_config.is_encoder_decoder:
                    output_src = se_model(**encoded_src, decoder_input_ids=encoded_src.input_ids)
                    output_synth = se_model(**encoded_synth, decoder_input_ids=encoded_synth.input_ids)
                else:
                    output_src = se_model(**encoded_src)
                    output_synth = se_model(**encoded_synth)

            embedding_src = torch.mean(output_src.last_hidden_state, dim=1)
            embedding_synth = torch.mean(output_synth.last_hidden_state, dim=1)

            similarity = torch.nn.functional.cosine_similarity(embedding_src, embedding_synth, dim=1)
            similarity_score = round(similarity.item(), 2)
        else:
            similarity_score = 0
        mean_similarity.append(similarity_score)

    return round(np.mean(mean_similarity), 2)

