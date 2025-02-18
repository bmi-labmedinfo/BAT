# BAT - Biomedical Augmentation for Text <img src="logo.png" width="60" style="vertical-align:middle;"/>

[![Contributors][contributors-shield]][contributors-url]
[![Watchers][watchers-shield]][watchers-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


[![Status][status-shield]][status-url] 

**Keywords**: *Data augmentation, Neuro-Symbolic AI, NLP, LLM, UMLS* </b>

------------------------------
A Toolkit for Biomedical Text Augmentation

<!-- PACKAGE OVERVIEW -->
# Package Overview
This Python package consists of a Neuro-Symbolic pipeline, blending *knowledge-driven* and *data-driven* approaches.

## Pipeline Components

**Knowledge-Based perturbation (*knowledge-driven*)**:
* **`Med-synonym replacement`**: Replaces medical terms with one of their formalized synonyms from structured domain knowledge (UMLS Metathesaurus).
* **`General synonym replacement`**: Replaces terms with one of their general-purpose synonyms from Wordnet.

**Transformer-Based perturbation (*data-driven*)**:
* **`Back-translation`**: Translates text into an intermediate language and then back into the original language using multilingual MT models.
* **`Contextual word prediction`**: Fills in masked single-token words within the input text based on the in-context predictions from BERT-based language models.
* **`Rephrasing`**: Rewrites text using the capabilities of LLMs.


<!-- REQUIREMENTS -->
### Requirements

1.  **Unified Medical Language System® (UMLS®) License**:
* Mandatory for using the `Med-Synonym Replacement` component.
* Optional for the `Synonym Replacement`.
2.  **LLM Functional Block**:
* A functional block with any preferred (open source or proprietary) LLM must be configured to use the `Rephrasing` component.
* Alternatively, you can use the default *gpt-4o-mini* model by providing your **personal API key**.

<!-- INSTALLATION -->
## Installation

1. Make sure you have the latest version of pip installed
   ```sh
   pip install --upgrade pip
    ```
2. Install `BiomedicalAugmentation-for-Text` through pip
    ```sh
    pip install --index-url https://test.pypi.org/simple/ --no-deps BiomedicalAugmentation-for-Text
    ```

<!-- USAGE EXAMPLES -->
## Usage

Here is a minimal example of how the BAT package can be invoked with `BiomedicalAugmentation-for-Text`.
1.  **Through the `AugmentedSample` class:** A compact and streamlined interface that integrates all components into a cohesive workflow.
```python
from bioTextAugPackage.init import *
import bioTextAugPackage.augmented_sample as aug_sample

config = Config()
input_text = "No lytic lesions are observed at the vertebral levels included in the scans. No signs of listhesis."
augmented_sample = aug_sample.AugmentedSample(config_params=config, technique_tag="TB-back_translation",
                                              src_data=input_text, src_lang="english", n_synth_data=5)
ans = augmented_sample.run()
```

2.  **By invoking individual functions:** Provides more control and flexibility to apply specific components independently.
```python
from bioTextAugPackage.init import *
import bioTextAugPackage.transformer_based_functions as tb
import  bioTextAugPackage.metrics as metrics

config = Config()
input_text = "No lytic lesions are observed at the vertebral levels included in the scans. No signs of listhesis."

src_lang = "en"
trg_lang = "fr"

mt_model_name1 = f"Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}"
mt_model1 = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name1)
mt_tokenizer1 = AutoTokenizer.from_pretrained(mt_model_name1)

mt_model_name2 = f"Helsinki-NLP/opus-mt-{trg_lang}-{src_lang}"
mt_model2 = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name2)
mt_tokenizer2 = AutoTokenizer.from_pretrained(mt_model_name2, clean_up_tokenization_spaces=True)

ans = tb.back_translation(src_text=input_text,
                          model1=mt_model1, tokenizer1=mt_tokenizer1,
                          model2=mt_model2, tokenizer2=mt_tokenizer2)

print(ans)
overlap_score = metrics.compute_overlap(synthetic_data=ans[0], src_data=input_text, tokenizer=config.base_tokenizer)
similarity_score = metrics.compute_similarity(synthetic_data=ans[0], src_data=input_text, se_model_name=config.se_model_name)
print(f"overlap_score: {overlap_score} - similarity_score: {similarity_score}")
```

A more extensive example, including advanced usage, can be found in [this notebook]().

<!-- CONTACTS AND USEFUL LINKS -->
## Contacts and Useful Links

*   **Repository maintainer**: Laura Bergomi  
    [![Email Badge][gmail-shield]][gmail-url] [![LinkedIn][linkedin-shield]][linkedin-url] 

*   **Project Link**: [https://github.com/bmi-labmedinfo/BAT](https://github.com/bmi-labmedinfo/BAT)

*   **Package Link**: [https://test.pypi.org/project/BiomedicalAugmentation-for-Text/](https://test.pypi.org/project/BiomedicalAugmentation-for-Text/)

<!-- LICENSE -->
## License

Distributed under MIT License. See `LICENSE` 

<!-- MARKDOWN LINKS -->
[logo]: logo.png
[contributors-shield]: https://img.shields.io/github/contributors/laurabergomi/BAT
[contributors-url]: https://github.com/bmi-labmedinfo/BAT/graphs/contributors
[status-shield]: https://img.shields.io/badge/Status-pre--release-blue
[status-url]: https://github.com/bmi-labmedinfo/BAT/releases
[forks-shield]: https://img.shields.io/github/forks/laurabergomi/BAT.svg
[forks-url]: https://github.com/bmi-labmedinfo/BAT/forks
[stars-shield]: https://img.shields.io/github/stars/laurabergomi/BAT.svg
[stars-url]: https://github.com/bmi-labmedinfo/BAT/stargazers
[issues-shield]: https://img.shields.io/github/issues/laurabergomi/BAT.svg
[issues-url]: https://github.com/bmi-labmedinfo/BAT/issues
[watchers-shield]: https://img.shields.io/github/watchers/laurabergomi/BAT.svg
[watchers-url]: https://github.com/bmi-labmedinfo/BAT/watchers
[license-shield]: https://img.shields.io/github/license/laurabergomi/BAT
[license-url]: https://github.com/bmi-labmedinfo/BAT/blob/main/LICENSE
[linkedin-shield]: 	https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff
[linkedin-url]: https://www.linkedin.com/in/laura-bergomi-628890293/
[gmail-shield]: https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white
[gmail-url]: mailto:laura.bergomi01@universitadipavia.it
