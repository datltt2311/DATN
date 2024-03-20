GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python getDetectionResults.py --bs 1 --cuda --mGPUs --image_dir /ext_data2/manhnh/MUSIC_dataset/frame/duet
#--bs 1 --cuda --mGPUs --image_dir /ext_data2/manhnh/MUSIC_dataset/frame/duet