import os
import shutil
import fire
from tqdm import tqdm


def split_files(source_dir, set1="set1", set2="set2"):
    # Directories to hold the split files
    set1_dir = os.path.join(source_dir, set1)
    set2_dir = os.path.join(source_dir, set2)

    # Create target directories if they don't exist
    os.makedirs(set1_dir, exist_ok=True)
    os.makedirs(set2_dir, exist_ok=True)

    # Function to get the prefix of the file
    def get_prefix(filename):
        return filename.split("_")[0]

    # Dictionary to keep track of which prefixes have been allocated
    prefix_set1 = {}
    prefix_set2 = {}
    set1_ctr = 0
    set2_ctr = 0
    # Loop through each file in the source directory
    for filename in tqdm(sorted(os.listdir(source_dir))):
        # Skip directories
        if os.path.isdir(os.path.join(source_dir, filename)) or not filename.endswith(
            ".pt"
        ):
            continue

        # Get the prefix of the file
        prefix = get_prefix(filename)
        # print(prefix)

        # if exists in set2
        if prefix in prefix_set2:
            target_dir = set1_dir
            if prefix in prefix_set1:
                prefix_set1[prefix].append(set1_dir)
            else:
                prefix_set1[prefix] = [set1_dir]
            set1_ctr += 1

        else:  # prefix not in set2
            target_dir = set2_dir
            if prefix in prefix_set2:
                prefix_set2[prefix].append(set2_dir)
            else:
                prefix_set2[prefix] = [set2_dir]
            set2_ctr += 1

        # # Move the file to the target directory
        shutil.move(
            os.path.join(source_dir, filename), os.path.join(target_dir, filename)
        )
    print("prefix_set1")
    print(set1_ctr)
    print(len(prefix_set1))
    # print(prefix_set1)

    print("prefix_set2")
    print(set2_ctr)
    print(len(prefix_set2))
    # print(prefix_set2)


if __name__ == "__main__":
    fire.Fire(split_files)
