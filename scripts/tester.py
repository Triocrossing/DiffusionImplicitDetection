import os
import argparse
import subprocess
from pathlib import Path

def create_symlink(target, link_name):
    try:
        os.symlink(target, link_name, target_is_directory=True)
    except FileExistsError:
        # Remove the existing link/file and create a new one
        os.remove(link_name)
        os.symlink(target, link_name, target_is_directory=True)

def read_symlink(link_name):
    return os.readlink(link_name)

def parse_test_dir(args):
    test_path = {
        "imagenet": "/home/xi/Work/DIRE/dataset/dire/dire/test/imagenet/imagenet",
        "lsun_bedroom": "/home/xi/Work/DIRE/dataset/dire/dire/test/lsun_bedroom/lsun_bedroom"
    }
    
    test_mode = {
      "png": "",
      "jpeg75": "_jpeg75",
    }
    if args.test_mode not in test_mode:
        raise ValueError(f"Unrecognized test mode: {args.test_mode}")
    if args.test_set not in test_path:
        raise ValueError(f"Unrecognized test set: {args.test_set}")
      
    final_path_real = Path(test_path[args.test_set]) / "real"
    final_path_fake = Path(test_path[args.test_set]) / str(args.test_method + test_mode[args.test_mode])
    # check existence of path
    assert final_path_real.exists() and final_path_fake.exists()
    return final_path_real, final_path_fake

def main():
    parser = argparse.ArgumentParser(description='Run tests with symbolic links, a specified checkpoint, and experiment name.')
    parser.add_argument('--ckpt',  type=str, help='Checkpoint file path')
    parser.add_argument('--exp_name',  type=str, default='default_experiment', help='Experiment name')
    parser.add_argument('--test_mode', type=str, default='test', help='Test mode')
    parser.add_argument('--test_set', type=str, help='Test set')
    parser.add_argument('--test_method', type=str, default='adm', help='Test method')
    args = parser.parse_args()
    
    path_real, path_fake = parse_test_dir(args)

    symlink_paths = {
        "1_fake": path_fake,
        "0_real": path_real
    }

    test_dir = f"data/DIRE/data/test/{args.test_set}"
    
    # Creating symbolic links
    for link_name, target in symlink_paths.items():
        create_symlink(target, os.path.join(test_dir, link_name))

    # Reading and printing the symbolic links
    print("test symlink:")
    for link_name in symlink_paths:
        print(read_symlink(os.path.join(test_dir, link_name)))


    # Running the Python script with arguments    
    command = ["python", "src/eval.py", 
               "data.testing_dir="+test_dir, 
               "data.training_dir="+test_dir, # can we do this?
               "data.validation_dir="+test_dir, 
               "ckpt_path="+ args.ckpt,
               "task_name="+"test",
               "tags="+"['"+args.test_set+",eval_"+args.test_method+"']",
               "data.batch_size=32",
               "logger.wandb.project=LaDDe_test",
               "logger.wandb.tags="+"['"+args.test_set+",eval_"+args.test_method+"']",
               "logger.wandb.name="+args.exp_name+"_test"+args.test_set+"_"+args.test_mode,]
    
    subprocess.run(command)
    


if __name__ == "__main__":
    main()
