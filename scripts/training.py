import argparse
import logging


def get_args():
    parser = argparse.ArgumentParser(
                    prog = "TUM-DLR-EO training script.",
                    description = "Trains the given model architecture.")
    parser.add_argument("--arch", required=True, help="Path of the architecture file.")
    parser.add_argument("--data", default="/dev/shm/demir2/hai_nasb_eo/data/", help="Path of the training data.")

    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    args = get_args()
    arch_path = args.arch
    