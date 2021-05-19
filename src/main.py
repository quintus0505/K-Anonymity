import click
from utils.user_config import load_data
from algorithms.Samarati import Samarati


################################################################################
# Settings
################################################################################
# @click.command()
# @click.argument('algorithm', type=click.Choice(['Samarati', 'Mondrian']))
# @click.option('--k', type=int, default=2, help='K-Anonymity parameter k (must be 2 <= nu).')
# @click.option('--maxsup', type=int, default=20,
#               help='K-Anonymity algorithm Samarati parameter maxSup (must be 2 <= nu).')
def main(algorithm='Samarati', k=5, maxsup=20):
    if algorithm == 'Samarati':
        alg = Samarati(algorithm_name='Samarati', k=k, maxsup=maxsup)
        alg.load_dataset()
        alg.initial_setting()
        alg.process()
        alg.save_data()



if __name__ == '__main__':
    main(algorithm='Samarati', k=3, maxsup=50)
