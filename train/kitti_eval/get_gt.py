import os
import numpy as np
import sys
import shutil


current_dir = os.getcwd()
results_dir = current_dir + "/result/"
new_gt_dir = current_dir + "/gt/"

ground_truth = "/home/kangning/Documents/Masterarbeit/frustum-pointnets/dataset/KITTI/object/training/label_2/"

file_names = os.listdir(results_dir)
ground_truth_names = os.listdir(ground_truth)


for file_name in file_names:
    if file_name in ground_truth_names:
        print(os.path.join(ground_truth,file_name))
        shutil.copy(os.path.join(ground_truth,file_name),new_gt_dir)
        print(file_name)
        
    
