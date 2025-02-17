import os
import argparse
from PIL import Image
from tqdm import tqdm
import pickle
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_pickle(pickle_path: str, if_cpu_unpickler):
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        if if_cpu_unpickler:
            data = CPU_Unpickler(f).load()
        else:
            data = pickle.load(f)
    return data


def convert_pkl_to_pth(source_root, target_root):
    for root, dirs, files in os.walk(source_root):
        for file in tqdm(files):
            if file.lower().endswith(".pkl"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_root)
                target_dir = os.path.join(target_root, relative_path)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                data = load_pickle(file_path, if_cpu_unpickler=False)
                _data = [data[0], data[-1]]
                output_file = os.path.splitext(file)[0] + ".pth"
                output_path = os.path.join(target_dir, output_file)
                torch.save(_data, os.path.join(target_dir, output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Convert pickle only saving first and last."
    )
    parser.add_argument("source", help="Source directory")
    parser.add_argument("target", help="Target directory")
    print

    args = parser.parse_args()

    convert_pkl_to_pth(args.source, args.target)


if __name__ == "__main__":
    main()
