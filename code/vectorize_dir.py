import argparse
import VectorModelBuilder

from os import listdir
from os.path import isfile, join

'''
Convenience script that produces a vector representation of all files in a
folder.
'''

DEFAULT_INDIR = '../corpora/noisy_parupa/'
DEFAULT_OUTDIR = '../vector_data/noisy_parupa/'

def vectorize_dir(indir, outdir, count_method, weighting):
    corpora = sorted([f for f in listdir(indir) if isfile(join(indir, f))])
    for f in corpora:
        full_path = join(indir, f)
        builder = VectorModelBuilder.VectorModelBuilder(
            full_path, count_method, weighting, outdir
        )
        builder.create_vector_model()
        builder.save_vector_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create vector embeddings for a directory of corpora files.'
    )
    parser.add_argument(
        '--indir', default=DEFAULT_INDIR, type=str,
        help='The directory of corpus files that will be vectorized.'
    )
    parser.add_argument(
        '--count_method', default=VectorModelBuilder.TRIGRAM, type=str,
        help='The method to use when created the context matrix.'
    )
    parser.add_argument(
        '--weighting', default=VectorModelBuilder.PPMI, type=str,
        help='The method to weight the raw counts.'
    )
    parser.add_argument(
        '--outdir', type=str, default=DEFAULT_OUTDIR,
        help='The directory to save the vector data in.'
    )

    args = parser.parse_args()
    vectorize_dir(args.indir, args.outdir, args.count_method, args.weighting)
