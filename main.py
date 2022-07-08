import os
import re
from email.mime import base

import numpy as np
import openai
import pandas as pd
from absl import app, flags, logging

import hlog
from utils import data

openai.ap_key = os.getenv("OPENAI_API_KEY")
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "prompt_file", default="inflection", help="Prompt file to use for the problem"
)
flags.DEFINE_string("language", default="itl", help="language for inflection task")
flags.DEFINE_integer("max_tokens", default=100, help="max LM generation length")
flags.DEFINE_multi_string("engine", default="text-davinci-002", help="GPT-3 models")
flags.DEFINE_integer("seed", default=1, help="seed for randomization")


def get_tarin_test_base(file):
    with open(f"prompts/{file}_train_base.txt") as handle:
        base_text = handle.read()
    with open(f"prompts/{file}_test_base.txt") as handle:
        test_text = handle.read()

    return base_text, test_text


def get_prompt(base_text, test_text, grouped_tags, test_item):
    prompt = []
    prompt_data = grouped_tags.sample(n=1)[:10]
    for index, row in prompt_data.iterrows():
        prompt.append(
            base_text.format(
                language="itl", inp=row["input"], tags=row["tags"], output=row["output"]
            )
        )

    prompt.append(
        test_text.format(language="itl", inp=test_item["input"], tags=test_item["tags"])
    )

    return "\n".join(prompt)


def main(argv):
    hlog.flags()
    np.random.seed(FLAGS.seed)
    train_path = f"data/{FLAGS.language}/{FLAGS.language}.train"
    test_path = f"data/{FLAGS.language}/{FLAGS.language}.test"

    train_text, test_text = get_tarin_test_base(FLAGS.prompt_file)

    data = pd.read_csv(
        train_path, sep="\t", header=None, names=["input", "output", "tags"]
    )
    hlog.value("data.head()", data.head())

    test_data = pd.read_csv(
        test_path, sep="\t", header=None, names=["input", "output", "tags"]
    )

    grouped_tags = data.groupby("tags")

    for index, test_item in test_data.iterrows():
        prompt = get_prompt(train_text, test_text, grouped_tags, test_item)
        print(prompt)
        break


if __name__ == "__main__":
    app.run(main)
