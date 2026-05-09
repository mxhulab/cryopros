import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

class FormatError(Exception):
    def __init__(self, message = "Not found header boundary, please check the format of inputted star file."):
        self.message = message
        super().__init__(self.message)

def generate_uniform_pose(file_path, out_path, particle_name):
    column_names = {
        '_rlnAngleRot': None,
        '_rlnAngleTilt': None,
        '_rlnAnglePsi': None,
        '_rlnImageName': None
    }

    headlines_buffer = []
    
    with open(file_path, 'r') as starfile:
        line_number = 0
        found_data = False
        for line in starfile:
            if "@" in line:
                headlines_n = line_number
                found_data = True
                break
            if line.startswith('_rln'):
                parts = line.split()
                column_name = parts[0]
                column_index = int(parts[-1].lstrip('#')) - 1
                if column_name in column_names:
                    column_names[column_name] = column_index
            headlines_buffer.append(line.strip())
            line_number += 1

        if not found_data:
            raise FormatError("No particle data found in the file.")

    if None in column_names.values():
        missing_columns = [key for key, value in column_names.items() if value is None]
        raise FormatError(f"Missing required columns in the STAR file: {missing_columns}")

    rot_n = column_names['_rlnAngleRot']
    tilt_n = column_names['_rlnAngleTilt']
    psi_n = column_names['_rlnAnglePsi']
    image_col = column_names['_rlnImageName']

    file = pd.read_table(file_path, sep="\\s+", header=None, skiprows=headlines_n)
    
    num_particles = len(file.iloc[:, image_col])
    num_digits = len(str(num_particles))
    file.iloc[:, image_col] = [f"{i+1:0{num_digits}d}@{particle_name}.mrcs" for i in range(num_particles)]

    pose_new = R.random(len(file)).as_euler("ZYZ", degrees=True)
    file.iloc[:, [rot_n, tilt_n, psi_n]] = pd.DataFrame(pose_new)

    df_headlines = pd.DataFrame(headlines_buffer)
    df_combined = pd.concat([df_headlines, file], ignore_index=True)
    df_combined.to_csv(out_path, sep="\t", index=False, header=False)
