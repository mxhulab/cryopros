import numpy as np
import pandas as pd
import argparse
import sys

class FormatError(Exception):
    def __init__(self, message = "Not found header boundary, please check the format of inputted star file."):
        self.message = message
        super().__init__(self.message)

def parse_argument():

    parser = argparse.ArgumentParser(description = 'Replacing poses in the input star file with poses sampled from a uniform distribution of spatial rotations.')

    parser.add_argument('--input' , type=str, help='input star file filename' , required = True)
    parser.add_argument('--output', type=str, help='output star file filename', required = True)

    # parser.add_argument('--boundary', type=int, help='Line number of header boundary(optional)', required=False)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    return parser.parse_args()

def main():

    
    args = parse_argument()

    # print(args)

    # 参数接收
    din = args.input
    dout = args.output

    # 默认列
    rot_n = 1
    tilt_n = 2
    psi_n = 3

    # 分界线识别与header信息暂存
    headlines_buffer = []
    with open(din, 'r') as starfile:
        line_number = 0
        found_data = False
        for line in starfile:
            if line_number > 100:
                raise FormatError()
            if line[0].isdigit() and "@" in line:
                headlines_n = line_number
                found_data = True
                break
            headlines_buffer.append(line.strip())
            line_number += 1
        if not found_data:
            raise FormatError()

    # 角度信息替换
    file = pd.read_table(
        din, sep="\\s+", header=None, skiprows=headlines_n
    )
    file_pose = file.iloc[:, [rot_n, tilt_n, psi_n]]
    pose = np.asarray(file_pose)
    file_new4 = file
    from scipy.spatial.transform import Rotation as R
    pose_new_4 = R.random(len(file)).as_euler("ZYZ", degrees=True)
    pose_new_4 = pd.DataFrame(pose_new_4)
    file_new4.iloc[:, [rot_n, tilt_n, psi_n]] = pose_new_4

    # header与角度信息整合保存
    df_headlines = pd.DataFrame(headlines_buffer)
    # df_headlines.to_csv("head_" + dout, sep="\t", index=False, header=False)
    df_combined = pd.concat([df_headlines, file_new4], ignore_index=True)
    df_combined.to_csv(dout, sep="\t", index=False, header=False)

if __name__ == "__main__":
    main()
