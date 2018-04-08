#20180323
import os
import pickle
import matplotlib.pyplot as plt
import numpy
np = numpy

def compareLearningCurves(param_vals,param_vals_con,FLAG_ALL=False,rate_list=[6400.0,6400.0],label_list=['3t','4t']):
    xEpoch = np.array(param_vals['iter'])/rate_list[0]
    y = param_vals

    plt.plot(xEpoch,y['train NLL (bits)'],label='train {}'.format(label_list[0]))
    if FLAG_ALL:
        plt.plot(xEpoch,y['valid NLL (bits)'],label='valid')
        plt.plot(xEpoch,y['test NLL (bits)'],label='test')

    xEpoch = np.array(param_vals_con['iter'])/rate_list[1]
    y = param_vals_con

    plt.plot(xEpoch,y['train NLL (bits)'],':',label='train {}'.format(label_list[1]))
    if FLAG_ALL:
        plt.plot(xEpoch,y['valid NLL (bits)'],':',label='valid')
        plt.plot(xEpoch,y['test NLL (bits)'],':',label='test')

    plt.ylabel('NLL(bits)')
    if sum(rate_list)==2: plt.xlabel('iter')
    else: plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return

def compareLearningCurves_path(path1,path2,rate_list=[6400.0,6400.0],label_list=['3t','4t']):
    with open(path1, 'rb') as f: param_vals = pickle.load(f)
    with open(path2, 'rb') as f: param_vals_pt2 = pickle.load(f)
    compareLearningCurves(param_vals,param_vals_pt2,FLAG_ALL=True,rate_list=rate_list,label_list=label_list)
    return

def getWeightEvolution(uc_para_expand,ucinit_para,fr_sz_list=[80,20,5]):
    delta_wav_sum,nb_wav_sum = 0.0,0.0
    delta_lab_sum,nb_lab_sum = 0.0,0.0
    for idx,fr_sz in enumerate(fr_sz_list):
        if idx==0: name = 'BigFrameLevel.GRU1.Step.Input.W0'
        elif len(fr_sz_list)==2: name = 'FrameLevel.InputExpand.W0'
        else: name = 'FrameLevel_{}.InputExpand.W0'.format(idx)
        w_uncon = uc_para_expand[name]
        w_ucinit = ucinit_para[name]
        delta = abs(w_uncon-w_ucinit)
        delta_wav = delta[:fr_sz,:]
        delta_lab = delta[fr_sz:,:]
        delta_wav_sum += numpy.sum(delta_wav)
        delta_lab_sum += numpy.sum(delta_lab)
        nb_wav_sum += numpy.prod(delta_wav.shape)
        nb_lab_sum += numpy.prod(delta_lab.shape)
    return delta_wav_sum/nb_wav_sum,delta_lab_sum/nb_lab_sum


def check_over_pretrain(uc_dir,ucinit_dir,uc_para_expand_dir,rate_list=[6400.0,6400.0],fr_sz_list=[80,20,5]):
    #1 check learing curve
    path1 = os.path.join(uc_dir,'train_log.pkl')
    path2 = os.path.join(ucinit_dir,'train_log.pkl')
    compareLearningCurves_path(path1,path2,rate_list=rate_list,label_list=['uc','ucinit'])
    
    #2 check weight evolution
    #2.1 load uc_para_expand
    with open(uc_para_expand_dir, 'rb') as f: uc_para_expand = pickle.load(f)
    
    #2.2 prepare lists for ucinit paras
    ucinit_params_dir = os.path.join(ucinit_dir,'params')
    para_list = os.listdir(ucinit_params_dir)[::5] #check for every 5 eps
    para_list = [p for p in para_list if '.pkl' in p]
    ep_list = []
    for para in para_list:
        ep = int(para.split('_')[1][1:])
        ep_list.append(ep)
    
    #2.3 load, compute, log delta
    delta_wav_list,delta_lab_list = [],[]
    for para in para_list:
        with open(os.path.join(ucinit_params_dir,para), 'rb') as f: ucinit_para = pickle.load(f)
        delta_wav,delta_lab = getWeightEvolution(uc_para_expand,ucinit_para,fr_sz_list=fr_sz_list)
        delta_wav_list.append(delta_wav)
        delta_lab_list.append(delta_lab)
        
    #2.4 plot
    plt.plot(ep_list,delta_wav_list,label='wav')
    plt.plot(ep_list,delta_lab_list,label='lab')
    plt.ylabel('mean of abs(delta) of weights')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    return

