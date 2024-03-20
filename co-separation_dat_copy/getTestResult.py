import subprocess
# iimpomport random
import os
list_file = []
#with open('test_solo_solo_250.txt') as f:
#with open('test_duet_duet.txt') as f:
# with open('test_duet_duet_100_yolo.txt') as f:
with open('test_solo_solo_250_yolo.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(" ")
        video_name1 = line[0]
        video_name2 = line[1].split('\n')[0]
        # line = line[0:-4]
        # print(line)
        # list_file.append(line)
        # cmd = ["python", "test.py", "--video1_name", video_name1,"--video2_name", video_name2,"--visual_pool", "conv1x1","--unet_num_layers", "7","--data_path", "/ext_data2/manhnh/MUSIC_dataset","--weights_visual", "/home/manhnguyen/new/co-separation/8_500000_checkpoint_dataD3/audioVisual/visual_best.pth","--weights_unet", "/home/manhnguyen/new/co-separation/8_500000_checkpoint_dataD3/audioVisual/unet_best.pth","--weights_classifier", "/home/manhnguyen/new/co-separation/8_500000_checkpoint_dataD3/audioVisual/classifier_best.pth","--num_of_object_detections_to_use", "5","--scene_path", "/home/manhnguyen/new/co-separation/iccv2019_co_separation/iccv2019_co_separation/ADE.h5","--output_dir_root", "resultsD3/duet_duet"]
        # change --weights_* to file best model .pth 
        cmd = ["python", "test.py", "--video1_name", video_name1,
               "--video2_name", video_name2,
               "--visual_pool", "conv1x1",
               "--unet_num_layers", "7",
               "--data_path", "/ext_data2/manhnh/MUSIC_dataset",
               "--weights_visual", "/ext_data2/manhnh/Checkpoints/checkpoint_only_pose_03072023/audioVisual/visual_best.pth",
               "--weights_unet", "/ext_data2/manhnh/Checkpoints/checkpoint_only_pose_03072023/audioVisual/unet_best.pth",
               "--weights_classifier", "/ext_data2/manhnh/Checkpoints/checkpoint_only_pose_03072023/audioVisual/classifier_best.pth",
               "--weights_vocal", "/ext_data2/manhnh/Checkpoints/checkpoint_only_pose_03072023/audioVisual/vocal_best.pth", 
               "--weights_facial", "/ext_data2/manhnh/Checkpoints/checkpoint_only_pose_03072023/audioVisual/facial_best.pth",
               "--num_of_object_detections_to_use", "5","--scene_path", "/home/manhnguyen/new/co-separation/iccv2019_co_separation/iccv2019_co_separation/ADE.h5",
               "--output_dir_root", "/ext_data2/manhnh/Results/result_only_pose_03072023/solo_solo"]
        subprocess.run(cmd)

