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
directory = '/scratch/qd212/datasets/speech/manuCutAlign_f32_norm_rmDC'#version2, replacing the line below
#directory = '/scratch/qd212/datasets'

FLAG_DATA_ALREADY = checkMakeDir(directory)


#1.2 if not already there, make dir and move data
if FLAG_DATA_ALREADY:
    print('data found on air')
else:
    print('data not found on air, moving ...')
    directory = '/scratch/qd212/datasets' #version2, newly added
    cmd = 'cp -r /home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/datasets/speech ' + directory
    output,err = runCMD(cmd)
    print('output: \n'+str(output))
    print('err: \n'+str(err))
    print('moving complete!')