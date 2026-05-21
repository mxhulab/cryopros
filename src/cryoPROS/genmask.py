import argparse
import sys
from . import __version__

def parse_argument():
    parser = argparse.ArgumentParser(description = 'Generating a volume mask for a given input volume and corresponding threshold.')

    parser.add_argument(
        '-v', '--version',
        action = 'version',
        version = f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--volume_path',
        required = True,
        help = 'input volume path'
    )
    parser.add_argument(
        '--result_path',
        required = True,
        help = 'output mask path'
    )
    parser.add_argument(
        '--threshold',
        type = float,
        required = True
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    args = parse_argument()

    import mrcfile
    import numpy as np

    with mrcfile.open(args.volume_path, permissive = True) as mrc:
        volume = mrc.data
        v_size = mrc.voxel_size

    mask = (volume > args.threshold).astype(np.uint16)
    with mrcfile.new(args.result_path, overwrite = True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = v_size

if __name__ == '__main__':
    main()
