import starfile
from scipy.spatial.transform import Rotation as R

def generate_uniform_pose(input_path, output_path, stack_path):
    star = starfile.read(input_path, always_dict = True)

    if len(star) == 1 and (0 in star or '' in star or 'images' in star):
        particles = list(star.values())[0]
    elif len(star) == 2 and ('optics' in star and 'particles' in star):
        particles = star['particles']
    else:
        raise RuntimeError('Invalid particle star')

    for key in ['rlnImageName', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']:
        if key not in particles:
            raise RuntimeError(f'Invalid particle star: missing {key}')

    n = len(particles)
    particles['rlnImageName'] = [f'{i + 1}@{stack_path}' for i in range(n)]
    pose_new = R.random(n).as_euler('ZYZ', degrees = True)
    particles['rlnAngleRot'] = pose_new[:, 0]
    particles['rlnAngleTilt'] = pose_new[:, 1]
    particles['rlnAnglePsi'] = pose_new[:, 2]

    starfile.write(star, output_path, overwrite = True)
