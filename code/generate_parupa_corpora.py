from HMM import HMM, START, END
from os.path import join

import argparse

'''
A script to generate one or more Parupa corpora at the specified noise level.
'''
DEFAULT_CORPORA_PER_LEVEL = 10
DEFAULT_CORPUS_SIZE = 50000
DEFAULT_OUTDIR = '../corpora/noisy_parupa/'

def generate_corpora(noise_levels, corpora_per_level, corpus_size, outdir):
    '''
    noise_levels: A list of noise levels, between 0 and 1, for each of which
                   one or more corpora will be generated.
    corpora_per_level: The number of corpora to generate at each noise level.
    corpus_size: The number of tokens per corpus.
    outdir: The directory to save the corpora in
    '''
    for noise_level in noise_levels:
        hmm = HMM()
        print('Generating data at noise level {}'.format(noise_level))

        # Add states
        hmm.add_state(1, 'Front High Consonant')
        hmm.add_state(2, 'Front Non-high Consonant')
        hmm.add_state(3, 'Front Vowel')
        hmm.add_state(4, 'Back High Consonant')
        hmm.add_state(5, 'Back Non-high Consonant')
        hmm.add_state(6, 'Back Vowel')
        hmm.add_state(7, 'Noisy consonant')
        hmm.add_state(8, 'Noisy vowel')

        # Vowel inventory:
        # front vowels: i, e
        # back vowels: u, o
        # transparent vowels: a
        # Consonant inventory:
        # Come before high vowels: t, k, p
        # Come before non-high vowels: d, g, b
        # Come word initially: p, b
        # Come before anything: r

        # Add transitions

        hmm.add_transition(
            START,
            7,
            [('p', 1/7),
             ('t', 1/7), 
             ('k', 1/7), 
             ('b', 1/7), 
             ('d', 1/7), 
             ('g', 1/7), 
             ('r', 1/7)],
            noise_level
        )

        hmm.add_transition(
            START,
            1, 
            [('p', 1)],
            (1 - noise_level) / 4
        )
        hmm.add_transition(
            START,
            2,
            [('b', 1)],
            (1 - noise_level) / 4
        )
        hmm.add_transition(
            START,
            4, 
            [('p', 1)],
            (1 - noise_level) / 4
        )
        hmm.add_transition(
            START,
            5,
            [('b', 1)],
            (1 - noise_level) / 4
        )

        hmm.add_transition(
            1,
            3,
            [('i', 1/2), 
             ('a', 1/2)],
            1
        )
        hmm.add_transition(
            2,
            3,
            [('e', 1/2), 
             ('a', 1/2)],
            1
        )
        hmm.add_transition(
            4,
            6,
            [('u', 1/2), 
             ('a', 1/2)],
            1
        )
        hmm.add_transition(
            5,
            6,
            [('o', 1/2), 
             ('a', 1/2)],
            1
        )

        hmm.add_transition(
            3,
            1,
            [('p', 1/4), 
             ('t', 1/4), 
             ('k', 1/4), 
             ('r', 1/4)],
            1/3
        )

        hmm.add_transition(
            3,
            2,
            [('b', 1/4), 
             ('d', 1/4), 
             ('g', 1/4), 
             ('r', 1/4)],
            1/3
        )

        hmm.add_transition(
            3,
            END,
            [('', 1)],
            1/3
        )

        hmm.add_transition(
            6,
            4,
            [('p', 1/4),
             ('t', 1/4),
             ('k', 1/4),
             ('r', 1/4)],
            1/3
        )

        hmm.add_transition(
            6,
            5,
            [('b', 1/4),
             ('d', 1/4),
             ('g', 1/4),
             ('r', 1/4)],
            1/3
        )

        hmm.add_transition(
            6,
            END,
            [('', 1)],
            1/3
        )

        hmm.add_transition(
            7,
            8,
            [
             ('a', 1/5),
             ('i', 1/5),
             ('e', 1/5),
             ('u', 1/5),
             ('o', 1/5)],
            1
        )

        hmm.add_transition(
            8,
            7,
            [
             ('p', 1/7),
             ('t', 1/7),
             ('k', 1/7),
             ('b', 1/7),
             ('d', 1/7),
             ('g', 1/7),
             ('r', 1/7)],
            2/3
        )
        hmm.add_transition(
            8,
            END,
            [('', 1)],
            1/3
        )

        for j in range(corpora_per_level):
            print('Generating corpus number {}...'.format(j))
            stringset = hmm.generate_stringset(corpus_size)
            
            outfile = join(
                outdir, 'noisy_parupa_{}_{}.txt'.format(
                    int(noise_level * 100), j
                )
            )
            with open(outfile, 'w') as f:
                for word in stringset:
                    print(' '.join(word), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates Parupa datasets.'
    )
    parser.add_argument(
        'noise_levels', type=float, nargs='+',
        help='The noise levels to generate corpora at. These should be'
                    'between 0 and 1, and passed in separated by spaces.'
    )
    parser.add_argument(
        '--corpora_per_level', type=int, default=DEFAULT_CORPORA_PER_LEVEL,
        help='The number of corpora to generate at each noise level.'
    )
    parser.add_argument(
        '--corpus_size', type=int, default=DEFAULT_CORPUS_SIZE,
        help='The number of tokens per corpus to generate.'
    )
    parser.add_argument(
        '--outdir', type=str, default=DEFAULT_OUTDIR,
        help='The directory to save output corpora in.'
    )
    args = parser.parse_args()

    generate_corpora(
        args.noise_levels, args.corpora_per_level, args.corpus_size,
        args.outdir
    )
