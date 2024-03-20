import subprocess
import os

list_file = []
arr = os.listdir("/home/manhnguyen/new/co-separation/hai_results/mix_hai/")
# print(arr)e
list_file = []
for dir in arr:
    # list_file.append(dir)
    print(dir)
    list_file.append("/home/manhnguyen/new/co-separation/hai_results/mix_hai/" + dir)
    # sub = os.listdir("./results/mixed/" + dir)
    cmd = ["python", "evaluateSeparation.py", "--results_dir","/home/manhnguyen/new/co-separation/hai_results/mix_hai/" + dir]
    subprocess.run(cmd)