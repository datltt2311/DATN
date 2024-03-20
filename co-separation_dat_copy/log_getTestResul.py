------------ Options -------------
audio_pool: maxpool
audio_sampling_rate: 11025
audio_window: 65535
batchSize: 32
binary_mask: False
checkpoints_dir: checkpoints_new_huy/
classifier_pool: maxpool
data_path: /ext_data2/manhnh/MUSIC_dataset
enable_data_augmentation: False
full_frame: False
gpu_ids: [0]
hdf5_path: /your_root/hdf5/MUSICDataset/soloduet
hop_size: 0.05
identity_feature_dim: 512
log_freq: True
mask_thresh: 0.5
mode: test
model: audioVisualMUSIC
nThreads: 16
name: audioVisual
num_of_object_detections_to_use: 5
number_of_classes: 15
output_dir_root: results_new_huy_44/duet
preserve_ratio: False
scene_path: /home/manhnguyen/new/co-separation/iccv2019_co_separation/iccv2019_co_separation/ADE.h5
seed: 0
spectrogram_type: magonly
stft_frame: 1022
stft_hop: 256
subtract_mean: True
unet_input_nc: 1
unet_ngf: 64
unet_num_layers: 7
unet_output_nc: 1
video1_name: /ext_data2/manhnh/MUSIC_dataset/yolo_top_detections/duet/top_detections_two_classes/B7vaxJhgCQE_5
video2_name: /ext_data2/manhnh/MUSIC_dataset/yolo_top_detections/duet/top_detections_two_classes/za84Zws-5gY_17
visual_pool: conv1x1
visualize_spectrogram: False
weighted_loss: False
weights_classifier: /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_44/audioVisual/classifier_best.pth
weights_facial: /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_44/audioVisual/facial_best.pth
weights_unet: /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_44/audioVisual/unet_best.pth
weights_visual: /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_44/audioVisual/visual_best.pth
weights_vocal: /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_44/audioVisual/vocal_best.pth
with_additional_scene_image: False
with_discriminator: False
with_frame_feature: False
with_silence_category: False
-------------- End ----------------
Loading weights for visual stream
Loading weights for UNet
Loading weights for audio classifier
Loading weights for vocal attributes analysis stream
Loading weights for facial attributes analysis stream
Traceback (most recent call last):
  File "test.py", line 297, in <module>
    main()
  File "test.py", line 196, in main
    outputs = model.forward(data)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 141, in forward
    return self.module(*inputs[0], **kwargs[0])
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/new/co-separation_copy_huy_copy/models/audioVisual_model.py", line 47, in forward
    visual_feature = self.net_visual(Variable(visuals, requires_grad=False))
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/new/co-separation_copy_huy_copy/models/networks.py", line 64, in forward
    x = self.feature_extraction(x)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/container.py", line 92, in forward
    input = module(input)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/container.py", line 92, in forward
    input = module(input)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torchvision/models/resnet.py", line 41, in forward
    out = self.conv1(x)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 320, in forward
    self.padding, self.dilation, self.groups)
KeyboardInterrupt
Traceback (most recent call last):
  File "getTestResult.py", line 17, in <module>
    subprocess.run(cmd)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/subprocess.py", line 474, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/subprocess.py", line 931, in communicate
    self.wait()
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/subprocess.py", line 990, in wait
    return self._wait(timeout=timeout)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/subprocess.py", line 1624, in _wait
    (pid, sts) = self._try_wait(0)
  File "/home/manhnguyen/anaconda3/envs/co_separation/lib/python3.7/subprocess.py", line 1582, in _try_wait
    (pid, sts) = os.waitpid(self.pid, wait_flags)
KeyboardInterrupt
