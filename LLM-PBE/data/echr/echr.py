# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets

from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger import Tagger
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted
from pii_leakage.utils.random import rnd_idx

from dataclasses import dataclass

@dataclass
class CustomECHRBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated
    shuffle_facts_seed: int = 42
    pseudonymize: bool = True


class CustomECHR(datasets.GeneratorBasedBuilder):
    """ A wrapper around the ECHR dataset that uses anonymization.  """

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the ECHR dataset."
    _TEXT = "text"

    _URLS = {
        "url": "LLM-PBE/echr-loc"
    }

    BUILDER_CONFIGS = [
        CustomECHRBuilder(name="undefended", sample_duplication_rate=1, version=VERSION,
                          description="undefended, private data"),
        CustomECHRBuilder(name="scrubbed", sample_duplication_rate=1, version=VERSION,
                          description="PII replaced with anon token")
    ]
    DEFAULT_CONFIG_NAME = "unprotected"

    def __init__(self, *args, **kwargs):
        self.df: pd.DataFrame = pd.DataFrame()
        ner_args = NERArgs(ner='flair',
                           ner_model="flair/ner-english-ontonotes-large",
                           anon_token="<MASK>",
                           anonymize=kwargs.setdefault("config_name", None) == "scrubbed")
        self._tagger: Tagger = TaggerFactory.from_ner_args(ner_args)
        print_dict_highlighted(ner_args.__dict__)
        super().__init__(*args, **kwargs)

    def _info(self):
        fea_dict = {self._TEXT: datasets.Value("string"),}
        if self.config.pseudonymize:
            fea_dict.update({entity_class: datasets.Value("string") 
               for entity_class in self._tagger.get_entity_classes()})
        features = datasets.Features(fea_dict)
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        self.df = load_dataset(self._URLS["url"])
        if 'test' in self.df.keys():
            self.df = concatenate_datasets([self.df["train"], self.df["test"], self.df["validation"]])

        # self.data = [item for sublist in self.df["facts"] for item in sublist]
        self.data = [item['prompt'] + item['target'] for item in self.df['train']]
        if self.config.shuffle_facts_seed > 0:
            self.data = [self.data[i] for i in rnd_idx(N=len(self.data), seed=self.config.shuffle_facts_seed)]

        return [
            datasets.SplitGenerator(  # use ~100k samples for the target model
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.45  # default: 0.45
                },
            ),
            datasets.SplitGenerator(  # use 10% of the training samples for test
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.45,
                    "end": 0.55  # default: 0.55
                },
            ),
            datasets.SplitGenerator(  # Use ~110k samples for shadow models
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": 0.55,
                    "end": 1.0  # default: 1.0
                },
            ),
        ]

    def _generate_examples(self, split: str, start: float, end: float):
        """ Given a start and stop location, tag all PII and generate the dataset.
        We use multi_gpu generation for improved speed.
        """
        start_pos, end_pos = int(len(self.data) * start), int(len(self.data) * end)
        print_highlighted(
            f"Length of data: {len(self.data)}. Scrubbing from {start_pos} to {end_pos} (Total={end_pos - start_pos}).")

        unique_identifier = start_pos
        for i, text in enumerate(self.data[start_pos:end_pos]):
            if self.config.pseudonymize:
                pseudonymized_text, piis = self._tagger.pseudonymize(text)
                # total_piis += len(piis)
                # if i% 100 == 0:
                #     print(f"Found {total_piis} piis")

                if i == 0:
                    print_highlighted(pseudonymized_text)

                pii_annotations = {k: ListPII() for k in self._tagger.get_entity_classes()}
                pii_annotations.update({k: v.dumps() for k, v in piis.group_by_class().items()})
            else:
                pseudonymized_text = text
                pii_annotations = {}
            
            for _ in range(self.config.sample_duplication_rate):
                yield f"{unique_identifier}", {
                    self._TEXT: pseudonymized_text,
                    **pii_annotations
                }
                unique_identifier += 1
