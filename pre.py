

import os
import shutil



if __name__ == "__main__":
    path = "D:/2 Projects/Data for BNC/ADHD-work3/rois_mat/"
    files = os.listdir(path)

    for i, f in enumerate(files):
        name = f[19:-12]  # f[:-33] / f[:-26]
        destination_folder = "D:/2 Projects/Data for BNC/ADHD-work3/data/{}/".format(name)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(path+f, destination_folder)