import os


arr = os.listdir('/ext_data2/manhnh/Results/result_visual_pose_05072023/solo_solo')
# print(arr)
list_file = []
for dir in arr:
    # list_file.append(dir)
    print(dir)
    list_file.append("/ext_data2/manhnh/Results/result_visual_pose_05072023/solo_solo/" + dir)
sdr = []
sdr_mixed = []
sdr_sdr_mixed  = []
sir = []
sar = []
with open('/ext_data2/manhnh/Results/result_visual_pose_05072023/solo_solo.txt','a') as f:
    for folder in list_file:
        try:
            output_file = open(os.path.join(folder, 'eval.txt'),'r')
            line = output_file.readlines()
            arr = line[0].split(" ")
            sdr.append(float(arr[0]))
            sdr_mixed.append(float(arr[1]))
            sdr_sdr_mixed.append(float(arr[2]))
            sir.append(float(arr[3]))
            sar.append(float(arr[4]))
            # print(line[0].split(" ")[0])
            f.write(folder.split('/')[6]+" | "+line[0]+'\n')
            output_file.close()
        except Exception as e:
            print(e)
    f.write(str(sum(sdr)/len(sdr))+" "+str(sum(sdr_mixed)/len(sdr_mixed))+" "+str(sum(sdr_sdr_mixed)/len(sdr_sdr_mixed))+" "+str(sum(sir)/len(sir))+" "+str(sum(sar)/len(sar)))
    f.close()