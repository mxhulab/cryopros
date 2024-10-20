import mrcfile
import numpy as np
import argparse
import sys

def parse_argument():

    parser = argparse.ArgumentParser(description = 'Generating a volume mask for a given input volume and corresponding threshold.')

    parser.add_argument('--volume_path', type=str, help='input volume path', required = True)
    parser.add_argument('--result_path', type=str, help='output mask path' , required = True)

    parser.add_argument('--threshold', type=float, required = True)

    if len(sys.argv) == 1:

        parser.print_help()
        exit()

    return parser.parse_args()

def main():

    args = parse_argument()

    dataroot = args.volume_path
    maskroot = args.result_path
    threshold = args.threshold

    with mrcfile.open(dataroot, permissive=True) as mrc:
        volume = mrc.data
        v_size = mrc.voxel_size

    mask = (volume > threshold).astype(np.uint8)

    with mrcfile.new(maskroot, overwrite=True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = v_size

if __name__ == '__main__':
    main()
