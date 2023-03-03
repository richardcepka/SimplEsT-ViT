import argparse
import os
import requests
import zipfile

from os.path import join as join_path


def parse_args() -> str:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        type=str,
        default="data",
        help="Root directory of dataset where directory tiny-imagenet-200 will be download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    root = parse_args().r

    if not os.path.exists(root): os.makedirs(root)

    # Download dataset
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = join_path(root, 'tiny-imagenet-200.zip')
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

    # Unzip dataset
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        def condition(s):
            """
                'tiny-imagenet-200/words.txt', 
                'tiny-imagenet-200/wnids.txt',
                'tiny-imagenet-200/train/*',
                'tiny-imagenet-200/val/*'
            """
            return 'test' not in s

        zip_ref.extractall(
            root, 
            members=list(filter(lambda s: condition(s), zip_ref.namelist()))
        ) 
    os.remove(filename)

    # Process training set
    train_dir = join_path(root, 'tiny-imagenet-200/train/')
    for label in os.listdir(train_dir):
        os.remove(join_path(train_dir, label, f"{label}_boxes.txt"))
        label_dir = join_path(train_dir, label, 'images')
        for filename in os.listdir(label_dir):
            old_path = join_path(label_dir, filename)
            new_path = old_path.replace('images/', '')
            os.rename(old_path, new_path)
        os.rmdir(label_dir)

    # Process val set
    val_dir = join_path(root, 'tiny-imagenet-200/val/')
    with open(join_path(val_dir, 'val_annotations.txt'), 'r') as f:
        for line in f:
            line_parts = line.strip().split('\t')
            filename = line_parts[0]
            label = line_parts[1]
            
            new_filename = filename.replace('val', label)
            old_path = join_path(val_dir, 'images', filename)
            new_dir = join_path(val_dir, label)
            new_path = join_path(new_dir, new_filename)
            os.makedirs(new_dir, exist_ok=True)
            os.rename(old_path, new_path)

    os.rmdir(join_path(val_dir, 'images'))
    os.remove(join_path(val_dir, 'val_annotations.txt'))
