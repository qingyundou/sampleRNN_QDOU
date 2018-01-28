import os
def checkDir(directory):
    if os.path.exists(directory):
        return True
    else:
        return False
    
import subprocess
def runCMD(cmd):
    df = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = df.communicate()
    return output,err

#1 remove data from air
#1.1 check directory
directory = '/scratch/qd212/datasets/'
#directory = 'asup/'

FLAG_DATA_ALREADY = checkDir(directory)


#1.2 if not already there, no need to remove
if FLAG_DATA_ALREADY:
    print('data found on air, removing ...')
    cmd = 'rm -r ' + directory
    output,err = runCMD(cmd)
    print('output: \n'+str(output))
    print('err: \n'+str(err))
    print('removing complete!')
else:
    print('data not found on air, no need to remove')