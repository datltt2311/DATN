import subprocess
# iimpomport random
import os
list_file = []
with open("/home/manhnguyen/new/co-separation/hai_list__mix.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.split("  ")
        video_name1 = line[0]
        video_name2 = line[1].split('\n')[0]
        cmd = ["python", "hai_test.py", "--video1_name", video_name1,"--video2_name", video_name2,"--visual_pool", "conv1x1","--unet_num_layers", "7","--data_path", "/ext_data2/manhnh/MUSIC_dataset","--weights_visual", "/home/manhnguyen/new/co-separation/hai_16_500000_checkpoint/audioVisual/visual_latest.pth","--weights_unet", "/home/manhnguyen/new/co-separation/hai_16_500000_checkpoint/audioVisual/unet_latest.pth","--weights_classifier", "/home/manhnguyen/new/co-separation/hai_16_500000_checkpoint/audioVisual/classifier_latest.pth","--num_of_object_detections_to_use", "5","--scene_path", "/home/manhnguyen/new/co-separation/iccv2019_co_separation/iccv2019_co_separation/ADE.h5","--output_dir_root", "hai_results/mix_hai/"]
        subprocess.run(cmd)
"""
python test.py --video1_name /ext_data2/manhnh/MUSIC_dataset/output_img/MUSIC_solo2/cello/Mff3pp-Fj2w_7 --video2_name /ext_data2/manhnh/MUSIC_dataset/output_img/MUSIC_solo2/saxophone/lm51ySOO71o_4 --visual_pool conv1x1 --unet_num_layers 7 --data_path /your_data_root/MUSIC_dataset/solo/ --weights_visual /ext_data2/students/hainghiem/co-separation/iccv2019_co_separation/iccv2019_co_separation/visual_latest.pth --weights_unet /ext_data2/students/hainghiem/co-separation/iccv2019_co_separation/iccv2019_co_separation/unet_latest.pth --weights_classifier /ext_data2/students/hainghiem/co-separation/iccv2019_co_separation/iccv2019_co_separation/classifier_latest.pth  --num_of_object_detections_to_use 5 --with_additional_scene_image --scene_path /ext_data2/students/hainghiem/co-separation/iccv2019_co_separation/iccv2019_co_separation/ADE.h5 --output_dir_root results_hai/
"""
#"/home/manhnguyen/new/co-separation/hai_16_500000_checkpoint/audioVisual/"