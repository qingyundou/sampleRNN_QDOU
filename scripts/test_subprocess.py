import subprocess
def runCMD(cmd):
    output = subprocess.check_output(cmd)
    return output

def runCMD(cmd):
    df = subprocess.Popen(["ls", "/home/non"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = df.communicate()
    return output,err

def runCMD(cmd):
    output = subprocess.check_output(cmd,stderr=subprocess.STDOUT)
    return output



cmd = 'ls'
#output,err = runCMD(cmd)

#print('1',output)
#print('2',err)
    
output = runCMD(cmd)
print('0',output)