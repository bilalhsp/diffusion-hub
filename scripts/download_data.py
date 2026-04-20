import time
import argparse
import logging
from omegaconf import OmegaConf

# local
from diffusion_hub import REPO_ROOT
from diffusion_hub.datasets import get_dataset, get_supported_datasets


def main(args):

    dataset_config = REPO_ROOT / 'configs' / 'data' / args.dataset / f'{args.dataset}.yaml'
    cfg = OmegaConf.load(dataset_config)

    dataset = get_dataset(cfg.name, **cfg.dataset_config, download=True)


if __name__ == '__main__':

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description='Script to download dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', dest='dataset', type=str, 
        action='store', default='ljspeech', required=True,
        choices=get_supported_datasets(),
        help='specify the dataset name to be downloaded.'
    )

    args = parser.parse_args()

    main(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run in total.")

