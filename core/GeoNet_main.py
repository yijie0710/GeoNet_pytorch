import argparse
import torch
from GeoNet_model import GeoNetModel
import yaml

def main():
    parser = argparse.ArgumentParser('description: GeoNet')
    parser.add_argument('--train',dest='train',action='store_true')
    parser.add_argument('--test',dest='test',action='store_true')
    parser.add_argument('--config',type=str)
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = yaml.load(f)
    
    geonet = GeoNetModel(config)

    if args.train:
        geonet.train()
    if args.test:
        geonet.test()

if __name__ == "__main__":
    main()
