import argparse
from omegaconf import OmegaConf
from gluefactory.datasets import get_dataset


def main(conf):
    dataset = get_dataset(conf.data.name)(conf.data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    args = parser.parse_args()
    conf = OmegaConf.load(args.conf)
    print(conf.data)
    main(conf)
