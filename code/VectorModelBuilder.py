import argparse
import nltk
import numpy as np

from math import log
from os import path

# Default files and directories
DEFAULT_OUTDIR = "../vector_data/"

# Counting methods
TRIGRAM = 'trigram'
BIGRAM = 'bigram'
TRIGRAM_BIGRAM = 'trigram_bigram'
NGRAM = 'ngram'

# Weighting methods
NONE = 'none'
PMI = 'pmi'
PPMI = 'ppmi'
PROBABILITY = 'probability'
CONDITIONAL_PROBABILITY = 'conditional_probability'

WORD_BOUNDARY = "#"

class VectorModelBuilder():
    """
    A class that takes in a dataset of words separated by newlines and 
    generates a vector embedding under the specified counting and
    weighting methods. Requires nltk and numpy to be installed.
    """
    def __init__(self, dataset, count_method=TRIGRAM,
                 weighting=PPMI, outdir=DEFAULT_OUTDIR, outfile=None, n=3):
        self.count_method = count_method
        self.n = n
        self.outfile = outfile
        self.outdir = outdir
        self.weighting = weighting

        self.sound_idx = []
        self.context_idx = []
        self.matrix = None
        
        self.weighting_functions = {
            PROBABILITY: self.matrix_to_probability,
            CONDITIONAL_PROBABILITY: self.matrix_to_conditional_probability,
            PPMI: self.matrix_to_PPMI,
            PMI: self.matrix_to_PMI,
            NONE: lambda: True
        }
        self.counting_functions = {
            TRIGRAM: self.count_trigrams,
            BIGRAM: self.count_bigrams,
            TRIGRAM_BIGRAM: self.count_trigram_bigrams,
            NGRAM: self.count_ngrams
        }

        self.preprocess_dataset(dataset)

    def preprocess_dataset(self, dataset):
        """
        Loads, removes duplicate words, and tokenizes the provided dataset.
        """
        self.dataset = dataset
        with open(self.dataset, 'r') as f:
            tokens = f.read()
        self.tokens = set([tuple(s.split(" ")) for s in tokens.split("\n") if s])
        self.tokens = [list(token) for token in self.tokens]

    def build_matrix(self):
        """
        Generates the matrix representing counts of sounds in contexts.
        """
        unique_sounds = set(
            [item for sublist in self.tokens for item in sublist]
        )
        self.sound_idx = sorted(list(unique_sounds))
        position_lists = self.counting_functions[self.count_method]()
        self.calculate_conditional_frequencies(position_lists)

    def count_bigrams(self):
        """
        Counts bigram occurrences in the token set.
        """
        bigrams = [
            x for token in self.tokens
            for x in nltk.bigrams([WORD_BOUNDARY] + token + [WORD_BOUNDARY])
        ]
        position_lists = [[], []]

        for a, b in bigrams:
            position_lists[0].append((b, a))
            position_lists[1].append((a, b))

        return [position_lists]

    def count_ngrams(self):
        ngrams = [
            x for token in self.tokens
            for x in nltk.ngrams(
                [WORD_BOUNDARY] * (self.n - 1) + token + [WORD_BOUNDARY] * (self.n - 1),
                self.n
            )
        ]

        position_lists = [[]] * self.n

        for gram in ngrams:
            for index, symbol in enumerate(gram):
                if symbol != WORD_BOUNDARY:
                    listgram = list(gram)
                    target = listgram.pop(index)
                    position_lists[index].append((tuple(listgram), target))

        return [position_lists]


    def count_trigrams(self):
        """
        Counts trigram occurrences in the token set.
        """
        trigrams = [
            x for token in self.tokens
            for x in nltk.trigrams(
                [WORD_BOUNDARY] * 2 + token + [WORD_BOUNDARY] * 2
            )
        ]
        position_lists = [[], [], []]

        for a, b, c in trigrams:
            if a != WORD_BOUNDARY:
                position_lists[0].append(((b, c), a))
            if b != WORD_BOUNDARY:
                position_lists[1].append(((a, c), b))
            if c != WORD_BOUNDARY:
                position_lists[2].append(((a, b), c))

        return [position_lists]

    def count_trigram_bigrams(self):
        """
        Counts trigrams and bigrams in the token set.
        """
        trigram_positions = self.count_trigrams()
        bigram_positions = self.count_bigrams()
        return bigram_positions + trigram_positions

    def calculate_conditional_frequencies(self, position_lists):
        """
        Calculates the conditional frequencies of each sound in the 
        provided list of n-gram tokens.
        """
        conditional_freqs = [
            [nltk.ConditionalFreqDist(l2) for l2 in l1] 
            for l1 in position_lists
        ]

        vec_len = sum(len(l) for c in conditional_freqs for l in c)
        num_sounds = len(self.sound_idx)
        self.matrix = np.zeros((num_sounds, vec_len))
        self.context_idx = []

        for sublist in conditional_freqs:
            for i, position_freqs in enumerate(sublist):
                for key, value in position_freqs.items():
                    context = list(key)
                    context.insert(i, '_')
                    context_label = '-'.join(context)
                    self.context_idx.append(context_label)
                    for sound, count in value.items():
                        if sound != WORD_BOUNDARY:
                            row = self.sound_idx.index(sound)
                            col = len(self.context_idx) - 1
                            self.matrix[row][col] = count

    def matrix_to_PPMI(self):
        """
        Weights the matrix of sound counts using PPMI.
        """
        self.matrix_to_PMI(ppmi=True)

    def matrix_to_PMI(self, ppmi=False):
        """
        Weights the matrix of counts using either PMI or PPMI.
        """
        weighted_matrix = np.zeros(self.matrix.shape)
        denominator = self.matrix.sum()

        for i in range(self.matrix.shape[0]):
            p_i = self.matrix[i].sum() / denominator
            for j in range(self.matrix.shape[1]):
                p_j = self.matrix[:, j].sum() / denominator
                p_ij = self.matrix[i][j] / denominator
                if not p_i or not p_j or not p_ij:
                    mi = 0
                else:
                    mi = log(p_ij / (p_i * p_j), 2)
                weighted_matrix[i][j] = max(mi, 0) if ppmi else mi
        self.matrix = weighted_matrix

    def matrix_to_conditional_probability(self):
        """
        Weights the matrix of counts using conditional probability.
        """
        weighted_matrix = np.zeros(self.matrix.shape)
        for i in range(self.matrix.shape[1]):
            col_sum = self.matrix[:, i].sum()
            for j in range(self.matrix.shape[0]):
                weighted_matrix[j, i] = self.matrix[j, i] / col_sum
        self.matrix = weighted_matrix

    def matrix_to_probability(self):
        """
        Weights the matrix of counts using probability.
        """
        weighted_matrix = np.zeros(self.matrix.shape)
        total_count = self.matrix.sum()
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                weighted_matrix[i, j] = self.matrix[i, j] / total_count
        self.matrix = weighted_matrix

    def create_vector_model(self):
        """
        Top-level function that counts sound occurrences and weights them
        using the specified methods.
        """
        print("Generating vector embedding for {}...".format(self.dataset))
        self.build_matrix()
        self.weighting_functions[self.weighting]()

    def save_vector_model(self):
        """
        Saves the generated vector embedding to three text files. The .data
        file contains the numeric vectors, the .sounds file contains the sound
        labels (row names) and the .contexts file contains the context names
        (column names).
        """
        if not self.outfile:
            base_components = [path.splitext(path.split(self.dataset)[1])[0]]
            base_components.append(self.count_method)
            base_components.append(self.weighting)
            base_str = '_'.join(base_components)
        else:
            base_str = self.outfile

        np.savetxt(path.join(
            self.outdir, '{}.data'.format(base_str)), self.matrix, fmt='%f'
        )
        with open(path.join(self.outdir, '{}.sounds'.format(base_str)), 'w') as f:
            print(' '.join(self.sound_idx), file=f)
        with open(path.join(self.outdir, '{}.contexts'.format(base_str)), 'w') as f:
            print(' '.join(self.context_idx), file=f)

if __name__ == "__main__":
    """
    Code for generating a vector embedding from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Create a vector space model for a phonological data set"
    )
    parser.add_argument(
        'dataset', type=str, help='The corpus to vectorize.'
    )
    parser.add_argument(
        '--count_method', default=TRIGRAM, type=str,
        help='The method to use when created the context matrix.'
    )
    parser.add_argument(
        '--weighting', default=PPMI, type=str,
        help='The method to weight the raw counts'
    )
    parser.add_argument(
        '--outfile', type=str, default=None,
        help='The filename to save the vector model under.'
    )
    parser.add_argument(
        '--outdir', type=str, default=DEFAULT_OUTDIR,
        help='The directory to save the vector data in.'
    )
    parser.add_argument(
        '--n', default=3, type=int,
        help='If count_method is "ngram", this specifies n.'
    )

    args = parser.parse_args()
    builder = VectorModelBuilder(
        args.dataset, args.count_method, args.weighting, args.outdir,
        args.outfile, args.n
    )
    builder.create_vector_model()
    builder.save_vector_model()
