import click
from utils.user_config import load_data
from algorithms.Samarati import Samarati
from algorithms.Mondrian import Mondrian


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('algorithm', type=click.Choice(['Samarati', 'Mondrian']))
@click.option('--k', type=int, default=2, help='K-Anonymity parameter k (must be 2 <= nu).')
@click.option('--maxsup', type=int, default=20,
              help='K-Anonymity algorithm Samarati parameter maxSup (must be 2 <= nu).')
def main(algorithm='Mondrian', k=5, maxsup=20):
    if algorithm == 'Samarati':
        alg = Samarati(algorithm_name='Samarati', k=k, maxsup=maxsup)
        alg.load_dataset()
        alg.initial_setting()
        alg.process()
        alg.save_data()
    else:
        alg = Mondrian(algorithm_name='Mondrian', k=k)
        alg.load_dataset()
        alg.initial_setting()
        alg.process()
        alg.compute_loss_metric()
        alg.save_data()


if __name__ == '__main__':
    main()
