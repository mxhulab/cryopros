import argparse
import sys
from . import __version__
from .logger import configure_logging, logger

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
    configure_logging(source='cryopros-prep')
    args = parse_argument()
    try:
        import mrcfile
        import numpy as np

        logger.info('Generating mask from %s with threshold %s', args.volume_path, args.threshold)
        with mrcfile.open(args.volume_path, permissive = True) as mrc:
            volume = mrc.data
            v_size = mrc.voxel_size

        mask = (volume > args.threshold).astype(np.uint16)
        with mrcfile.new(args.result_path, overwrite = True) as mrc:
            mrc.set_data(mask)
            mrc.voxel_size = v_size
        logger.info('Wrote mask to %s', args.result_path)
    except Exception:
        logger.exception('CryoPROS mask generation failed')
        raise

if __name__ == '__main__':
    main()
