import numpy
np = numpy

def checkMakeDir(directory):
    if os.path.exists(directory):
        return True
    else:
        os.makedirs(directory)
        return False

import scipy.io.wavfile
import os
BITRATE = 16000
def write_audio_file(name, data, save_dir):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(
                    os.path.join(save_dir+name+'.wav'),
                    BITRATE,
                    data)
        
def readWav(dirFile,rate=3.0518e-05):
    speech_wav = scipy.io.wavfile.read(dirFile)
    data = speech_wav[1]
    data = data.astype('float32')
    #data *= rate
    #data /= 32768.0
    return data


#1.1 build idx_list, for cut
dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/lab/utt_lab/speech_test_utt_lab.npy'
data = np.load(dirFile)
length_list = []
for utt in data:
    length_list.append(utt.shape[0])
idx_list = np.cumsum(length_list)

#1.2 build name_list, for save
f = open('/home/dawna/tts/qd212/mphilproj/data/speech/file_id_list.scp','r')
tmp = f.readlines()
fileList = [line[:-1] for line in tmp]
f.close()
name_list = fileList[-72:]

#2 loop over exps, cut and save
rate = 80
results_dir = '/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/results_3t_gen/'
for exp_dir in os.listdir(results_dir):
    print exp_dir
    #2.1 get waveform to be cut
    data_wav_dir = results_dir + exp_dir + '/samples/'
    data_wav_list = []
    for filename in os.listdir(data_wav_dir):
        if 'sample_8s' in filename:
            dirFile = data_wav_dir + filename
            tmp = readWav(dirFile)
            data_wav_list.append(tmp)
    data_wav_array = np.array(data_wav_list)
    tmp = np.reshape(data_wav_array,(1,-1))
    data_wav = tmp[0]

    #2.2 cut waveform and save
    save_dir = data_wav_dir + 'format/'
    checkMakeDir(save_dir)
    for i,l in enumerate(idx_list):
        print i,
        end = l*rate
        if i==0: start = 0
        else: start = idx_list[i-1]*rate
        if end>len(data_wav):
            print 'over'
            break
        else:
            #cut
            utt_wav = data_wav[start:end]
            #save
            name = name_list[i]
            print name
            write_audio_file(name, utt_wav, save_dir)