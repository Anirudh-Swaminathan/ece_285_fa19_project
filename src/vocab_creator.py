#!/usr/bin/env python3

# imports go here
from pycocotools.coco import COCO
from collections import Counter
import nltk
import re


class VocabCreate(object):
    """Class to create the vocabulary for our project"""

    def __init__(self, data_pth):
        """Constructor
        """
        self.data_pth = data_pth
        self.data = None
        self.vocab_list = Counter()

    def load_data(self):
        """Function to load the data"""
        self.data = COCO(self.data_pth)

    def list_words(self):
        """Creates a list of words"""
        # iterate through every caption
        cap_ids = self.data.anns.keys()
        id = 47
        cap = str(self.data.anns[id]["caption"])
        print(cap)
        cap = "Hi! i am her.i am not?"
        op = re.sub(r'[^a-zA-Z0-9 ]+', '', cap)
        print(cap)
        print(op)
        word = nltk.tokenize.word_tokenize(op.lower())
        print(word)
        """
        for id in cap_ids:
            cap = str(self.data.anns[id]["caption"])
            self.vocab_list.update(cap.lower())
        """


def main():
    vocab = VocabCreate("../datasets/COCO/annotations/captions_train2014.json")
    vocab.load_data()
    vocab.list_words()


if __name__ == "__main__":
    main()
