import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

from optogenetic_holography.arg_parser import ArgParser


if __name__ == '__main__':
    args = ArgParser().parse_all_args()
    print(args)

