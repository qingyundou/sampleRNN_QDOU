import os
def checkMakeDir(directory):
    if os.path.exists(directory):
        return True
    else:
        os.makedirs(directory)
        return False
    
import subprocess
def runCMD(cmd):
    df = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = df.communicate()
    return output,err

#1 copy data to air
#1.1 check directory

src_all_dataset_dir = '/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/datasets/speech/'
src_all_dataset_list = os.listdir(src_all_dataset_dir)

tgt_all_dataset_dir = '/scratch/qd212/datasets/speech/'
FLAG_DATA_ALREADY = checkMakeDir(tgt_all_dataset_dir)
tgt_all_dataset_list = os.listdir(tgt_all_dataset_dir)



#1.2 if not already there, make dir and move data
#loop over src, if not found in tgt, copy
for dataset in src_all_dataset_list:
    if dataset=='16k_resil_Lesley':
        t = os.path.join(tgt_all_dataset_dir,dataset)
        cmd = 'rm -r {}'.format(t)
        print('cleaning Lesley with cmd: '+cmd)
        output,err = runCMD(cmd)
        print('output: '+str(output))
        print('err: '+str(err))
        print('cleaning complete!')
    if dataset not in tgt_all_dataset_list:
        s = os.path.join(src_all_dataset_dir,dataset)
        t = os.path.join(tgt_all_dataset_dir,dataset)
        checkMakeDir(t)
        cmd = 'cp -r {s}/* {t}/'.format(s=s,t=t)
        print('moving with cmd: '+cmd)
        output,err = runCMD(cmd)
        print('output: '+str(output))
        print('err: '+str(err))
        print('moving complete!')