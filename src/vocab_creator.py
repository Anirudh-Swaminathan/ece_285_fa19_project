#!/usr/bin/env python3

# imports go here
from pycocotools.coco import COCO
from collections import Counter
import re


class VocabCreate(object):
    """Class to create the vocabulary for our project"""

    def __init__(self, data_pth):
        """Constructor
        """
        self.data_pth = data_pth
        self.data = None
        self.dict = None
        self.vocab_list = Counter()
        self.load_data()
        self.frequency_list()
        self.remove_lowest(thresh=5)
        # print(len(self.vocab_list.keys()))
        vals = [self.vocab_list[v] for v in self.dict]
        # print(set(vals), len(vals))

    def load_data(self):
        """Function to load the data"""
        self.data = COCO(self.data_pth)

    def frequency_list(self):
        """Creates a list of words"""
        # iterate through every caption
        cap_ids = self.data.anns.keys()
        for id in cap_ids:
            cap = str(self.data.anns[id]["caption"])
            clean_cap = re.sub(r'^[a-zA-Z0-9 ]+', '', cap)
            word_list = clean_cap.lower().strip().split()
            self.vocab_list.update(word_list)

    def __len__(self):
        return len(self.dict)

    def remove_lowest(self, thresh):
        """
        Only keeps those words whose frequency is greater than threshold
        :argument thresh minimum frequency to keep in the vocabulary
        """
        ret = [w for w in self.vocab_list.keys() if self.vocab_list[w] >= thresh]
        self.dict = ret


def main():
    vocab = VocabCreate("../datasets/COCO/annotations/captions_train2014.json")
    print("Loaded successfully!")
    print("There are ", len(vocab), " unique words with frequency >=5 in the dataset")


if __name__ == "__main__":
    main()
