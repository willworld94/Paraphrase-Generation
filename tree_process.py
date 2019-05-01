from nltk.parse import CoreNLPParser
import os
parser = CoreNLPParser(url='http://localhost:9000')

def tree_process(source, output):
    #     source_file = open("data/tgt-train.txt", "r")
    source_file = open(source, "r")
    source_lines = source_file.readlines()
    #     with open('data/tgt-train-tree.txt', 'a') as f1:
    with open(output, 'a') as f1:
        for line in source_lines:
            out = str(list(parser.raw_parse(line)))
            out = out.replace("[Tree", '')
            out = out.replace('[', '')
            out = out.replace(']', '')
            out = out.replace("Tree", '')
            out = out.replace(",", '')
            out = out.replace("'", '')
            result = tknzr.tokenize(out)
            output = ' '.join([token for token in result])
            f1.write(output + os.linesep)
