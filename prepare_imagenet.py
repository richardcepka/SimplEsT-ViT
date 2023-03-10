import argparse
import os
from os.path import join as join_path

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import ImageFolder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/imagenet",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default="data/ffcv_imagenet",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--jpeg_quality",
        type=float,
        default=90.,
    )
    parser.add_argument(
        "--write_mode",
        type=str,
        default="smart",
    )
    return parser.parse_args()


def main(data_dir, write_path, num_workers, 
        chunk_size, max_resolution, 
        jpeg_quality, write_mode):
    
    my_dataset = ImageFolder(root=data_dir)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.write_dir): os.makedirs(args.write_dir)
    main(
        data_dir=join_path(args.data_dir, args.split),
        write_path=join_path(args.write_dir, args.split),
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        max_resolution=args.max_resolution,
        jpeg_quality=args.jpeg_quality,
        write_mode=args.write_mode
    )