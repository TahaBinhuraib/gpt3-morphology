import codecs
import re


def read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for line in lines:
        line = line.strip().split("\t")
        if line:
            inputs.append(list(line[0]))
            outputs.append(list(line[1]))
            tags.append(re.split("\W+", line[2]))

    return inputs, outputs, tags
