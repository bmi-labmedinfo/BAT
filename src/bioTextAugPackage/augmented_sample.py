"""
Main class to invoke functions from the Biomedical Augmentation for Text package.

Classes:
    AugmentedSample
"""
from .init import *
from .knowledge_based_functions import *
from .metrics import *
from .transformer_based_functions import *
from typing import Literal


class AugmentedSample:
    """
        A class used to manage Biomedical Augmentation for Text Package calls.
        ...
        Attributes
        ----------
        config : object
            config params
        techniques : dict
            list of text-augmentation techniques and corresponding call
        technique_tag : str
            specific text-augmentation technique to perform
        src_lang : str
            source data language code (it,en,fr,es)
        src_data : str
            source data to augment
        n_synth_data : int
            max number of new samples as output
        similarity_threshold : float
            default minimum similarity score
        customizable_params : dict
            additional customizable parameters
        new_synth_data : list
            new synthetic samples
        overlap_score : list
            new samples overlap scores
        similarity_score : list
            new samples similarity scores
        synonyms_dict = dict
            synonym dictionary

        Methods
        -------
        _random_languages():
            random selection of languages for back translation.
        _create_synonyms_dict():
            creation of a synonym dictionary for medical terms.
        synonym_replacement():
            synonym_replacement technique call.
        med_synonym_replacement():
            med_synonym_replacement technique call.
        back_translation():
            back_translation technique call.
        masked_lm():
            masked_lm technique call.
        llm_rephrasing():
            llm_rephrasing technique call.
        run():
            run the text augmentation technique selected and return the new synthetic data.
    """

    def __init__(self, config_params: Config,
                 method: Literal["KB-synonym_replacement", "KB-med_synonym_replacement", "TB-back_translation", "TB-masked_lm", "TB-llm_rephrasing"],
                 src_lang: str, src_data: str, n_synth_data: int = 1,
                 **kwargs):
        """
        Augmented sample class.
        :param config_params: config params
        :param method: text-augmentation technique
        :param src_lang: source data language (italian, english, spanish, french)
        :param src_data: source data to augment
        :param n_synth_data: max number of new samples as output
        :param kwargs: additional customizable parameters (default:
            - custom_intermediate_language: List[str] = random -> back-translation custom intermediate languages code
            - medical_field: str = None -> medical field context for llm_rephrasing generation
            - multilingual: bool = True -> let default models set to multilingual
            - verbose: bool = False -> print intermediate results)
        """
        self.config = config_params
        self.techniques = {
            "KB-synonym_replacement": self.synonym_replacement,
            "KB-med_synonym_replacement": self.med_synonym_replacement,
            "TB-back_translation": self.back_translation,
            "TB-masked_lm": self.masked_lm,
            "TB-llm_rephrasing": self.llm_rephrasing
        }
        self.technique_tag = method
        if self.technique_tag not in self.techniques.keys():
            print(
                f"{StringColors.WARNING}> The technique tag inserted is not valid! Valid options are: "
                f"{self.techniques.keys()}!")
            sys.exit(1)

        self.src_lang = lookup_language(language=src_lang)
        self.src_data = src_data
        self.n_synth_data = n_synth_data
        self.similarity_threshold = 0.8

        self.customizable_params = kwargs
        if kwargs.get("custom_intermediate_language") is None:
            self.customizable_params["custom_intermediate_language"] = self._random_languages()
        else:
            self.customizable_params["custom_intermediate_language"] = \
                self.customizable_params["custom_intermediate_language"][:self.n_synth_data]
        if kwargs.get('medical_field') is None:
            self.customizable_params['medical_field'] = None
        if kwargs.get('multilingual') is None:
            self.customizable_params['multilingual'] = True
        if not self.customizable_params['multilingual']:
            self.config.set_lang_based_models(src_lang)
        if kwargs.get('verbose') is None:
            self.customizable_params['verbose'] = False

        self.new_synth_data = []
        self.overlap_score = []
        self.similarity_score = []
        self.synonyms_dict = {}

    def _random_languages(self):
        """
        Set n=n_synth_data random languages for back-translation, excluding the src_lang.
        If the source_lang is not "english", English will be used by default as one of the translation languages.
        :return: List[str]: list of languages
        """
        intermediate_languages: List[str] = ["it", "es", "de", "fr", "ar"]
        if self.src_lang in intermediate_languages:
            intermediate_languages.remove(self.src_lang)
        if self.src_lang == "it":
            intermediate_languages.remove("fr")
        if self.src_lang == "fr":
            intermediate_languages.remove("it")
        random_langs = random.sample(intermediate_languages, min(self.n_synth_data, len(intermediate_languages)))
        if self.src_lang != "en":
            random_langs += ["en"]
            random_langs = random_langs[-self.n_synth_data:]
        return random_langs

    def _create_synonyms_dict(self):
        """
        Creation of a synonym dictionary for medical terms.
        :return: dict: synonym dictionary
        """
        synonyms_dict = synonyms_dict_definition(src_data=self.src_data, src_tokenizer=self.config.base_tokenizer,
                                                 src_lang=self.src_lang, api_key=self.config.umls_api_key,
                                                 verbose=self.customizable_params['verbose'])
        return synonyms_dict

    def synonym_replacement(self):
        """
        Synonym_replacement technique call.
        If the umls_api_key is provided (optional), the medical terms are excluded from the replacement.
        :return: List[str]: new data
        """
        if self.config.umls_api_key is not None and self.config.umls_api_key != "":
            self.synonyms_dict = self._create_synonyms_dict()
        new_data = synonym_replacement(text=self.src_data, src_lang=self.src_lang,
                                       med_synonyms_dict=self.synonyms_dict, n_new_data=self.n_synth_data)
        return new_data

    def med_synonym_replacement(self):
        """
        Med_synonym_replacement technique call.
        It is mandatory to provide an umls_api_key.
        :return: List[str]: new data
        """
        if not self.config.umls_api_key:
            print(f"{StringColors.WARNING}> UMLS API_KEY must be inserted!")
            sys.exit(1)
        self.synonyms_dict = self._create_synonyms_dict()
        new_data = med_synonym_replacement(text=self.src_data, synonyms_dict=self.synonyms_dict,
                                           n_new_data=self.n_synth_data)
        return new_data

    def back_translation(self):
        """
        Back_translation technique call.
        A new data is created for each language.
        :return: List[str]: new data
        """
        new_text = []

        for trg_lang in tqdm(self.customizable_params["custom_intermediate_language"],
                             desc="Intermediate languages translation"):
            mt_model_name1 = f"Helsinki-NLP/opus-mt-{self.src_lang}-{trg_lang}"
            mt_model1 = MarianMTModel.from_pretrained(mt_model_name1)
            mt_tokenizer1 = MarianTokenizer.from_pretrained(mt_model_name1, clean_up_tokenization_spaces=True)

            mt_model_name2 = f"Helsinki-NLP/opus-mt-{trg_lang}-{self.src_lang}"
            mt_model2 = MarianMTModel.from_pretrained(mt_model_name2)
            mt_tokenizer2 = MarianTokenizer.from_pretrained(mt_model_name2, clean_up_tokenization_spaces=True)

            new_data = back_translation(src_text=self.src_data, model1=mt_model1, tokenizer1=mt_tokenizer1,
                                        model2=mt_model2, tokenizer2=mt_tokenizer2)
            new_text.append(new_data)

        return new_text

    def masked_lm(self):
        """
        Masked_lm technique call.
        :return: List[str]: new data
        """
        new_data = masked_augmentation(src_text=self.src_data, ml_model_name=self.config.bert_model_name,
                                       ml_tokenizer=self.config.bert_tokenizer, n_new_data=self.n_synth_data)
        return new_data

    def llm_rephrasing(self):
        """
        Llm_rephrasing technique call.
        It is mandatory to provide an llm_api_key.
        :return: List[str]: new data
        """
        if not self.config.llm_caller and not self.config.llm_api_key:
            print(f"{StringColors.WARNING}> LLM API_KEY must be inserted!")
            sys.exit(1)

        new_data = llm_generation(src_text=self.src_data, n_new_data=self.n_synth_data, src_lang=self.src_lang,
                                  tokenizer=self.config.base_tokenizer,
                                  api_key=self.config.llm_api_key, caller_func=self.config.llm_caller,
                                  medical_field=self.customizable_params['medical_field'],
                                  verbose=self.customizable_params['verbose'])
        return new_data

    def run(self):
        """
        Run the text augmentation technique selected and return the new synthetic data.
        :return:
            - src_data: source data -> List[str]
            - synth_data: new synthetic data -> List[str]
            - overlap_score: average overlap score -> List[str]
            - similarity_score: average sentence embedding similarity score -> List[str]
        """
        # ============== selected pipeline ==============
        new_data = self.techniques[self.technique_tag]()  # -> List[str]
        # ===============================================

        for elem in tqdm(new_data, desc="Computing metrics"):
            overlap_value = compute_overlap(synthetic_data=elem, src_data=self.src_data,
                                            tokenizer=self.config.base_tokenizer)
            similarity_value = compute_similarity(synthetic_data=elem, src_data=self.src_data,
                                                  se_model_name=self.config.se_model_name)
            self.new_synth_data.append(elem)
            self.overlap_score.append(overlap_value)
            self.similarity_score.append(similarity_value)

        output_dict = {
            "tag": [self.technique_tag] * len(new_data) if self.technique_tag != "TB-back_translation" else [
                f"{self.technique_tag}-{lookup_language(lang_code=el)}" for el in
                self.customizable_params["custom_intermediate_language"]],
            "src_data": [self.src_data] * len(new_data),
            "synth_data": self.new_synth_data,
            "overlap_score": self.overlap_score,
            "similarity_score": self.similarity_score
        }
        return pd.DataFrame.from_dict(output_dict).drop_duplicates(subset=["src_data", "synth_data"])
