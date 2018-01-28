"""
RNN Audio Generation Model

Three-tier model, Quantized input
For more info:
$ python three_tier.py -h

How-to-run example:
sampleRNN$ pwd
/u/mehris/sampleRNN


sampleRNN$ \
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u \
models/three_tier/three_tier.py --exp AXIS1 --seq_len 512 --big_frame_size 8 \
--frame_size 2 --weight_norm True --emb_size 64 --skip_conn False --dim 32 \
--n_rnn 2 --rnn_type LSTM --learn_h0 False --q_levels 16 --q_type linear \
--batch_size 128 --which_set MUSIC

To resume add ` --resume` to the END of the EXACTLY above line. You can run the
resume code as many time as possible, depending on the TRAIN_MODE.
(folder name, file name, flags, their order, and the values are important)
"""
from time import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time()

import os, sys, glob
sys.path.insert(1, os.getcwd())
import argparse
import itertools

import numpy
numpy.random.seed(123)
np = numpy
import random
random.seed(123)

import theano
import theano.tensor as T
import theano.ifelse
import lasagne
import scipy.io.wavfile

import lib

import pdb

LEARNING_RATE = 0.001



### Parsing passed args/hyperparameters ###
def get_args():
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    # No default value here. Indicate every single arguement.
    parser = argparse.ArgumentParser(
        description='three_tier.py\nNo default value! Indicate every argument.')

    # TODO: Fix the descriptions
    # Hyperparameter arguements:
    parser.add_argument('--exp', help='Experiment name',
            type=str, required=False, default='_')
    parser.add_argument('--seq_len', help='How many samples to include in each\
            Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--big_frame_size', help='How many samples per big frame',\
            type=check_positive, required=True)
    parser.add_argument('--frame_size', help='How many samples per frame',\
            type=check_positive, required=True)
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization\
            to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--emb_size', help='Size of embedding layer (> 0)',
            type=check_positive, required=True)  # different than two_tier
    parser.add_argument('--skip_conn', help='Add skip connections to RNN',
            type=t_or_f, required=True)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--learn_h0', help='Whether to learn the initial state of RNN',\
            type=t_or_f, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of\
            audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale, a-law-companding,\
            or mu-law compandig. With mu-/a-law quantization level shoud be set as 256',\
            choices=['linear', 'a-law', 'mu-law'], required=True)
    parser.add_argument('--which_set', help='ONOM, BLIZZ, MUSIC, or HUCK, or SPEECH',
            choices=['ONOM', 'BLIZZ', 'MUSIC', 'HUCK', 'SPEECH'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[1, 20, 64, 128, 256], required=True)

    parser.add_argument('--debug', help='Debug mode', required=False, default=False, action='store_true')
    parser.add_argument('--resume', help='Resume the same model from the last\
            checkpoint. Order of params are important. [for now]',\
            required=False, default=False, action='store_true')
    
    parser.add_argument('--n_big_rnn', help='For tier3, Number of layers in the stacked RNN',\
            type=check_positive, choices=xrange(1,6), required=False, default=0)
    
    parser.add_argument('--rmzero', help='remove q_zero, start from real data',\
            required=False, default=False, action='store_true')
    parser.add_argument('--normed', help='normalize data on corpus level',\
            required=False, default=False, action='store_true')
    parser.add_argument('--grid', help='use data on air',\
            required=False, default=False, action='store_true')
    
    parser.add_argument('--quantlab', help='quantize labels',\
            required=False, default=False, action='store_true')
    
    parser.add_argument('--ep', help='which epoch to generate',\
            type=str,required=False, default=20)
    

    args = parser.parse_args()

    # NEW
    # Create tag for this experiment based on passed args
    tag = reduce(lambda a, b: a+b, sys.argv).replace('--resume', '').replace('/', '-').replace('--', '-').replace('True', 'T').replace('False', 'F')
    tag += '-lr'+str(LEARNING_RATE)
    print "Created experiment tag for these args:"
    print tag
    
    #deal with pb - dir name too long
    #option1
    #tag = reduce(lambda a, b: a+b, sys.argv[:-4]).replace('--resume', '').replace('/', '-').replace('--', '-').replace('True', 'T').replace('False', 'F')
    #option2
    tag = tag.replace('-which_setSPEECH','').replace('size','sz').replace('frame','fr').replace('batch','bch').replace('--grid', '')
    
    #tag += '-lr'+str(LEARNING_RATE)
    
    #maxTag = 200
    #if len(tag)>maxTag: tag = tag[:maxTag]

    return args, tag

args, tag = get_args()

SEQ_LEN = args.seq_len # How many samples to include in each truncated BPTT pass
#print "------------------previous SEQ_LEN:", SEQ_LEN
# TODO: test incremental training
#SEQ_LEN = 512 + 256
#print "---------------------------new SEQ_LEN:", SEQ_LEN
BIG_FRAME_SIZE = args.big_frame_size # how many samples per big frame
FRAME_SIZE = args.frame_size # How many samples per frame
OVERLAP = BIG_FRAME_SIZE
WEIGHT_NORM = args.weight_norm
EMB_SIZE = args.emb_size
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
BIG_DIM = DIM # Dimensionality for the slowest level.
N_RNN = args.n_rnn # How many RNNs to stack in the frame-level model

if args.n_big_rnn==0:
    N_BIG_RNN = N_RNN # how many RNNs to stack in the big-frame-level model
else:
    N_BIG_RNN = args.n_big_rnn
#pdb.set_trace()

RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1
LEARN_H0 = args.learn_h0
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
Q_TYPE = args.q_type # log- or linear-scale
WHICH_SET = args.which_set
BATCH_SIZE = args.batch_size
RESUME = args.resume
assert SEQ_LEN % BIG_FRAME_SIZE == 0,\
    'seq_len should be divisible by big_frame_size'
assert BIG_FRAME_SIZE % FRAME_SIZE == 0,\
    'big_frame_size should be divisible by frame_size'
N_FRAMES = SEQ_LEN / FRAME_SIZE # Number of frames in each truncated BPTT pass

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')

    
#epoch index of the parameter for synthesis
EXP_NAME = args.exp
EP_IDX = 'e'+args.ep
    
###set FLAGS for options
flag_dict = {}
flag_dict['RMZERO'] = args.rmzero
flag_dict['NORMED_ALRDY'] = args.normed
flag_dict['GRID'] = args.grid
flag_dict['QUANTLAB'] = args.quantlab

FLAG_QUANTLAB = flag_dict['QUANTLAB']

# Fixed hyperparams
GRAD_CLIP = 1 # Elementwise grad clip threshold
BITRATE = 16000

# Other constants
#TRAIN_MODE = 'iters' # To use PRINT_ITERS and STOP_ITERS
TRAIN_MODE = 'time' # To use PRINT_TIME and STOP_TIME
#TRAIN_MODE = 'time-iters'
# To use PRINT_TIME for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
#TRAIN_MODE = 'iters-time'
# To use PRINT_ITERS for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
PRINT_ITERS = 10000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 100000 # Stop after this many iterations
PRINT_TIME = 12*60*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*24*3 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
N_SEQS = 10  # Number of samples to generate every time monitoring.
RESULTS_DIR = 'results_3t_gen'
FOLDER_PREFIX = os.path.join(RESULTS_DIR, tag)
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

LAB_SIZE = 80 #one label covers 80 points on waveform
LAB_PERIOD = float(0.005) #one label covers 0.005s ~ 200Hz
LAB_DIM = 601
UP_RATE = LAB_SIZE/FRAME_SIZE

epoch_str = 'epoch'
iter_str = 'iter'
lowest_valid_str = 'lowest valid cost'
corresp_test_str = 'correponding test cost'
train_nll_str, valid_nll_str, test_nll_str = \
    'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'

if args.debug:
    import warnings
    warnings.warn('----------RUNNING IN DEBUG MODE----------')
    TRAIN_MODE = 'time'
    PRINT_TIME = 100
    STOP_TIME = 3000
    STOP_ITERS = 1000

### Create directories ###
#   FOLDER_PREFIX: root, contains:
#       log.txt, __note.txt, train_log.pkl, train_log.png [, model_settings.txt]
#   FOLDER_PREFIX/params: saves all checkpoint params as pkl
#   FOLDER_PREFIX/samples: keeps all checkpoint samples as wav
#   FOLDER_PREFIX/best: keeps the best parameters, samples, ...
if not os.path.exists(FOLDER_PREFIX):
    os.makedirs(FOLDER_PREFIX)
PARAMS_PATH = os.path.join(FOLDER_PREFIX, 'params')
if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)
SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples')
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)
BEST_PATH = os.path.join(FOLDER_PREFIX, 'best')
if not os.path.exists(BEST_PATH):
    os.makedirs(BEST_PATH)

lib.print_model_settings(locals(), path=FOLDER_PREFIX, sys_arg=True)


### Import the data_feeder ###
# Handling WHICH_SET
if WHICH_SET == 'ONOM':
    from datasets.dataset import onom_train_feed_epoch as train_feeder
    from datasets.dataset import onom_valid_feed_epoch as valid_feeder
    from datasets.dataset import onom_test_feed_epoch  as test_feeder
elif WHICH_SET == 'BLIZZ':
    from datasets.dataset import blizz_train_feed_epoch as train_feeder
    from datasets.dataset import blizz_valid_feed_epoch as valid_feeder
    from datasets.dataset import blizz_test_feed_epoch  as test_feeder
elif WHICH_SET == 'MUSIC':
    from datasets.dataset import music_train_feed_epoch as train_feeder
    from datasets.dataset import music_valid_feed_epoch as valid_feeder
    from datasets.dataset import music_test_feed_epoch  as test_feeder
elif WHICH_SET == 'HUCK':
    from datasets.dataset import huck_train_feed_epoch as train_feeder
    from datasets.dataset import huck_valid_feed_epoch as valid_feeder
    from datasets.dataset import huck_test_feed_epoch  as test_feeder
elif WHICH_SET == 'SPEECH':
    from datasets.dataset_con import speech_train_feed_epoch as train_feeder
    from datasets.dataset_con import speech_valid_feed_epoch as valid_feeder
    from datasets.dataset_con import speech_test_feed_epoch  as test_feeder

    
def get_lab_big(seqs_lab):
    seqs_lab_big = seqs_lab[:,::BIG_FRAME_SIZE/FRAME_SIZE,:]
    return seqs_lab_big


def load_data(data_feeder):
    """
    Helper function to deal with interface of different datasets.
    `data_feeder` should be `train_feeder`, `valid_feeder`, or `test_feeder`.
    """
    return data_feeder(FRAME_SIZE,
                       BATCH_SIZE,
                       SEQ_LEN,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)
def load_data_gen(data_feeder,SEQ_LEN_gen):
    return data_feeder(FRAME_SIZE,
                       BATCH_SIZE,
                       SEQ_LEN_gen,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)
print('----got to def---')
### Creating computation graph ###
def big_frame_level_rnn(input_sequences, input_sequences_lab_big, h0, reset):
    """
    input_sequences.shape: (batch size, n big frames * BIG_FRAME_SIZE)
    h0.shape:              (batch size, N_BIG_RNN, BIG_DIM)
    reset.shape:           ()
    output[0].shape:       (batch size, n frames, DIM)
    output[1].shape:       same as h0.shape
    output[2].shape:       (batch size, seq len, Q_LEVELS)
    """
    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] // BIG_FRAME_SIZE,
        BIG_FRAME_SIZE
    ))

    if FLAG_QUANTLAB:
        frames = T.concatenate([frames, input_sequences_lab_big], axis=2)
        # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
        # (a reasonable range to pass as inputs to the RNN)
        frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
        frames *= lib.floatX(2)
    else:
        input_sequences_lab_big *= lib.floatX(2) # 0< data <2
        input_sequences_lab_big -= lib.floatX(1) # -1< data <1
        input_sequences_lab_big *= lib.floatX(2) # -2< data <2
        
        frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
        frames *= lib.floatX(2)
        frames = T.concatenate([frames, input_sequences_lab_big], axis=2)

    # Initial state of RNNs
    learned_h0 = lib.param(
        'BigFrameLevel.h0',
        numpy.zeros((N_BIG_RNN, H0_MULT*BIG_DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_BIG_RNN, H0_MULT*BIG_DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    #---debug---
    #pdb.set_trace()
    #---debug---
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('BigFrameLevel.GRU',
                                                   N_BIG_RNN,
                                                   BIG_FRAME_SIZE+LAB_DIM,
                                                   BIG_DIM,
                                                   frames,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('BigFrameLevel.LSTM',
                                                    N_BIG_RNN,
                                                    BIG_FRAME_SIZE+LAB_DIM,
                                                    BIG_DIM,
                                                    frames,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(
        'BigFrameLevel.Output',
        BIG_DIM,
        DIM * BIG_FRAME_SIZE / FRAME_SIZE,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    output = output.reshape((output.shape[0], output.shape[1] * BIG_FRAME_SIZE / FRAME_SIZE, DIM))

    independent_preds = lib.ops.Linear(
        'BigFrameLevel.IndependentPreds',
        BIG_DIM,
        Q_LEVELS * BIG_FRAME_SIZE,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    independent_preds = independent_preds.reshape((independent_preds.shape[0], independent_preds.shape[1] * BIG_FRAME_SIZE, Q_LEVELS))

    return (output, last_hidden, independent_preds)

def frame_level_rnn(input_sequences, input_sequences_lab, other_input, h0, reset):
    """
    input_sequences.shape: (batch size, n frames * FRAME_SIZE)
    other_input.shape:     (batch size, n frames, DIM)
    h0.shape:              (batch size, N_RNN, DIM)
    reset.shape:           ()
    output.shape:          (batch size, n frames * FRAME_SIZE, DIM)
    """
    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] // FRAME_SIZE,
        FRAME_SIZE
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    frames *= lib.floatX(2)
    
    if FLAG_QUANTLAB:
        frames = T.concatenate([frames, input_sequences_lab], axis=2)

        # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
        # (a reasonable range to pass as inputs to the RNN)
        frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
        frames *= lib.floatX(2)
        
    else:
        input_sequences_lab *= lib.floatX(2) # 0< data <2
        input_sequences_lab -= lib.floatX(1) # -1< data <1
        input_sequences_lab *= lib.floatX(2) # -2< data <2
        
        frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
        frames *= lib.floatX(2)
        
        frames = T.concatenate([frames, input_sequences_lab], axis=2)

    gru_input = lib.ops.Linear(
        'FrameLevel.InputExpand',
        FRAME_SIZE+LAB_DIM,
        DIM,
        frames,
        initialization='he',
        weightnorm=WEIGHT_NORM,
        ) + other_input

    # Initial state of RNNs
    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    #learned_h0 = T.patternbroadcast(learned_h0, [False] * learned_h0.ndim)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('FrameLevel.GRU',
                                                   N_RNN,
                                                   DIM,
                                                   DIM,
                                                   gru_input,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('FrameLevel.LSTM',
                                                    N_RNN,
                                                    DIM,
                                                    DIM,
                                                    gru_input,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(
        'FrameLevel.Output',
        DIM,
        FRAME_SIZE * DIM,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    output = output.reshape((output.shape[0], output.shape[1] * FRAME_SIZE, DIM))

    return (output, last_hidden)

def sample_level_predictor(frame_level_outputs, prev_samples):
    """
    frame_level_outputs.shape: (batch size, DIM)
    prev_samples.shape:        (batch size, FRAME_SIZE)
    output.shape:              (batch size, Q_LEVELS)
    """
    # Handling EMB_SIZE
    if EMB_SIZE == 0:  # no support for one-hot in three_tier and one_tier.
        prev_samples = lib.ops.T_one_hot(prev_samples, Q_LEVELS)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE, Q_LEVELS)
        last_out_shape = Q_LEVELS
    elif EMB_SIZE > 0:
        prev_samples = lib.ops.Embedding(
            'SampleLevel.Embedding',
            Q_LEVELS,
            EMB_SIZE,
            prev_samples)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE, EMB_SIZE), f32
        last_out_shape = EMB_SIZE
    else:
        raise ValueError('EMB_SIZE cannot be negative.')

    prev_samples = prev_samples.reshape((-1, FRAME_SIZE * last_out_shape))

    out = lib.ops.Linear(
        'SampleLevel.L1_PrevSamples',
        FRAME_SIZE * last_out_shape,
        DIM,
        prev_samples,
        biases=False,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    out += frame_level_outputs
    # out = T.nnet.relu(out)  # commented out to be similar to two_tier

    out = lib.ops.Linear('SampleLevel.L2',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # L3
    out = lib.ops.Linear('SampleLevel.L3',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # Output
    # We apply the softmax later
    out = lib.ops.Linear('SampleLevel.Output',
                         DIM,
                         Q_LEVELS,
                         out,
                         weightnorm=WEIGHT_NORM)
    return out

print('----got to T var---')
sequences   = T.imatrix('sequences')
h0          = T.tensor3('h0')
big_h0      = T.tensor3('big_h0')
reset       = T.iscalar('reset')
mask        = T.matrix('mask')
if FLAG_QUANTLAB:
    print('REMINDER: lab is quantized')
    sequences_lab      = T.itensor3('sequences_lab')
    sequences_lab_big      = T.itensor3('sequences_lab_big')
else:
    print('REMINDER: lab is NOT quantized')
    sequences_lab      = T.tensor3('sequences_lab')
    sequences_lab_big      = T.tensor3('sequences_lab_big')
    
if args.debug:
    # Solely for debugging purposes.
    # Maybe I should set the compute_test_value=warn from here.
    sequences.tag.test_value = numpy.zeros((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='int32')
    h0.tag.test_value = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')
    big_h0.tag.test_value = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*BIG_DIM), dtype='float32')
    reset.tag.test_value = numpy.array(1, dtype='int32')
    mask.tag.test_value = numpy.ones((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='float32')


big_input_sequences = sequences[:, :-BIG_FRAME_SIZE]
input_sequences = sequences[:, BIG_FRAME_SIZE-FRAME_SIZE:-FRAME_SIZE]
target_sequences = sequences[:, BIG_FRAME_SIZE:]

target_mask = mask[:, BIG_FRAME_SIZE:]

#---debug---
#pdb.set_trace()
#---debug---
big_frame_level_outputs, new_big_h0, big_frame_independent_preds = big_frame_level_rnn(big_input_sequences, sequences_lab_big, big_h0, reset)

frame_level_outputs, new_h0 = frame_level_rnn(input_sequences, sequences_lab, big_frame_level_outputs, h0, reset)

prev_samples = sequences[:, BIG_FRAME_SIZE-FRAME_SIZE:-1]
prev_samples = prev_samples.reshape((1, BATCH_SIZE, 1, -1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1, FRAME_SIZE), neib_step=(1, 1), mode='valid')
prev_samples = prev_samples.reshape((BATCH_SIZE * SEQ_LEN, FRAME_SIZE))

sample_level_outputs = sample_level_predictor(
    frame_level_outputs.reshape((BATCH_SIZE * SEQ_LEN, DIM)),
    prev_samples
)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs),
    target_sequences.flatten()
)
cost = cost.reshape(target_sequences.shape)
cost = cost * target_mask
# Don't use these lines; could end up with NaN
# Specially at the end of audio files where mask is
# all zero for some of the shorter files in mini-batch.
#cost = cost.sum(axis=1) / target_mask.sum(axis=1)
#cost = cost.mean(axis=0)

# Use this one instead.
cost = cost.sum()
cost = cost / target_mask.sum()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
# log_2(e) = 1.44269504089
cost = cost * lib.floatX(numpy.log2(numpy.e))

ip_cost = lib.floatX(numpy.log2(numpy.e)) * T.nnet.categorical_crossentropy(
    T.nnet.softmax(big_frame_independent_preds.reshape((-1, Q_LEVELS))),
    target_sequences.flatten()
)
ip_cost = ip_cost.reshape(target_sequences.shape)
ip_cost = ip_cost * target_mask
ip_cost = ip_cost.sum()
ip_cost = ip_cost / target_mask.sum()

### Getting the params, grads, updates, and Theano functions ###
#params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
#ip_params = lib.get_params(ip_cost, lambda x: hasattr(x, 'param') and x.param==True\
#    and 'BigFrameLevel' in x.name)
#other_params = [p for p in params if p not in ip_params]
#params = ip_params + other_params
#lib.print_params_info(params, path=FOLDER_PREFIX)
#
#grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
#grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]
#
#updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)

###########
all_params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
ip_params = lib.get_params(ip_cost, lambda x: hasattr(x, 'param') and x.param==True\
    and 'BigFrameLevel' in x.name)
other_params = [p for p in all_params if p not in ip_params]
all_params = ip_params + other_params
lib.print_params_info(ip_params, path=FOLDER_PREFIX)
lib.print_params_info(other_params, path=FOLDER_PREFIX)
lib.print_params_info(all_params, path=FOLDER_PREFIX)

ip_grads = T.grad(ip_cost, wrt=ip_params, disconnected_inputs='warn')
ip_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in ip_grads]

other_grads = T.grad(cost, wrt=other_params, disconnected_inputs='warn')
other_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in other_grads]

grads = T.grad(cost, wrt=all_params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

#---debug---
#pdb.set_trace()
#---debug---

ip_updates = lasagne.updates.adam(ip_grads, ip_params)
other_updates = lasagne.updates.adam(other_grads, other_params)
updates = lasagne.updates.adam(grads, all_params)

print('----got to fn---')
# Training function(s)
"""
ip_train_fn = theano.function(
    #[sequences, sequences_lab, sequences_lab_big, big_h0, reset, mask],
    [sequences, sequences_lab_big, big_h0, reset, mask],
    [ip_cost, new_big_h0],
    updates=ip_updates,
    on_unused_input='warn'
)

other_train_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, big_h0, h0, reset, mask],
    [cost, new_big_h0, new_h0],
    updates=other_updates,
    on_unused_input='warn'
)
"""
train_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, big_h0, h0, reset, mask],
    [cost, new_big_h0, new_h0],
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function, hence no updates
"""
ip_test_fn = theano.function(
    #[sequences, sequences_lab, sequences_lab_big, big_h0, reset, mask],
    [sequences, sequences_lab_big, big_h0, reset, mask],
    [ip_cost, new_big_h0],
    on_unused_input='warn'
)

other_test_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, big_h0, h0, reset, mask],
    [cost, new_big_h0, new_h0],
    on_unused_input='warn'
)
"""
test_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, big_h0, h0, reset, mask],
    [cost, new_big_h0, new_h0],
    on_unused_input='warn'
)

# Sampling at big frame level
big_frame_level_generate_fn = theano.function(
    [sequences, sequences_lab_big, big_h0, reset],
    big_frame_level_rnn(sequences, sequences_lab_big, big_h0, reset)[0:2],
    on_unused_input='warn'
)

# Sampling at frame level
big_frame_level_outputs = T.matrix('big_frame_level_outputs')
frame_level_generate_fn = theano.function(
    [sequences, sequences_lab, big_frame_level_outputs, h0, reset],
    frame_level_rnn(sequences, sequences_lab, big_frame_level_outputs.dimshuffle(0,'x',1), h0, reset),
    on_unused_input='warn'
)

# Sampling at audio sample level
frame_level_outputs = T.matrix('frame_level_outputs')
prev_samples        = T.imatrix('prev_samples')
sample_level_generate_fn = theano.function(
    [frame_level_outputs, prev_samples],
    lib.ops.softmax_and_sample(
        sample_level_predictor(
            frame_level_outputs,
            prev_samples
        )
    ),
    on_unused_input='warn'
)

# Uniform [-0.5, 0.5) for half of initial state for generated samples
# to study the behaviour of the model and also to introduce some diversity
# to samples in a simple way. [it's disabled]

FLAG_USETRAIN_WHENTEST = False
def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= numpy.mean(data)
        data /= numpy.absolute(data).max()
        data /= 2.0
        scipy.io.wavfile.write(
                    os.path.join(SAMPLES_PATH, name+'.wav'),
                    BITRATE,
                    data)

    total_time = time()
    # Generate N_SEQS' sample files, each 5 seconds long
    N_SECS = 8
    LENGTH = N_SECS*BITRATE if not args.debug else 100

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')

    if FLAG_USETRAIN_WHENTEST:
        print('')
        print('REMINDER: using training data for test')
        print('')
        testData_feeder = load_data_gen(train_feeder,LENGTH)
    else:
        testData_feeder = load_data_gen(test_feeder,LENGTH)
    mini_batch = testData_feeder.next()
    tmp, _, _, seqs_lab = mini_batch
    samples_lab = seqs_lab[:N_SEQS]
    
    if flag_dict['RMZERO']:
        samples[:, :BIG_FRAME_SIZE] = tmp[:N_SEQS, :BIG_FRAME_SIZE]
    else:
        samples[:, :BIG_FRAME_SIZE] = Q_ZERO
    
    samples_lab_big = get_lab_big(samples_lab)
    
    # First half zero, others fixed random at each checkpoint
    big_h0 = numpy.zeros(
            (N_SEQS, N_BIG_RNN, H0_MULT*BIG_DIM),
            dtype='float32'
    )

    h0 = numpy.zeros(
            (N_SEQS, N_RNN, H0_MULT*DIM),
            dtype='float32'
    )

    big_frame_level_outputs = None
    frame_level_outputs = None

    for t in xrange(BIG_FRAME_SIZE, LENGTH):

        if t % BIG_FRAME_SIZE == 0:
            tmp = samples_lab_big[:,(t-BIG_FRAME_SIZE)//BIG_FRAME_SIZE,:]
            tmp = tmp.reshape(tmp.shape[0],1,tmp.shape[1])
            
            big_frame_level_outputs, big_h0 = big_frame_level_generate_fn(
                samples[:, t-BIG_FRAME_SIZE:t],
                tmp,
                big_h0,
                numpy.int32(t == BIG_FRAME_SIZE)
            )

        if t % FRAME_SIZE == 0:
            tmp = samples_lab[:,(t-FRAME_SIZE)//FRAME_SIZE,:]
            tmp = tmp.reshape(tmp.shape[0],1,tmp.shape[1])
            
            frame_level_outputs, h0 = frame_level_generate_fn(
                samples[:, t-FRAME_SIZE:t],
                tmp,
                big_frame_level_outputs[:, (t / FRAME_SIZE) % (BIG_FRAME_SIZE / FRAME_SIZE)],
                h0,
                numpy.int32(t == BIG_FRAME_SIZE)
            )

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE],
            samples[:, t-FRAME_SIZE:t]
        )

    total_time = time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, N_SECS, total_time)
    print log,

    for i in xrange(N_SEQS):
        samp = samples[i]
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        write_audio_file("sample_{}_{}".format(tag, i), samp)

def monitor(data_feeder):
    """
    Cost and time of test_fn on a given dataset section.
    Pass only one of `valid_feeder` or `test_feeder`.
    Don't pass `train_feed`.

    :returns:
        Mean cost over the input dataset (data_feeder)
        Total time spent
    """
    _total_time = time()
    _h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')
    _big_h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*BIG_DIM), dtype='float32')
    _costs = []
    _data_feeder = load_data(data_feeder)
    for _seqs, _reset, _mask, _seqs_lab in _data_feeder:
        _seqs_lab_big = get_lab_big(_seqs_lab)
        _cost, _big_h0, _h0 = test_fn(_seqs, _seqs_lab, _seqs_lab_big, _big_h0, _h0, _reset, _mask)
        _costs.append(_cost)

    return numpy.mean(_costs), time() - _total_time

print "Wall clock time spent before training started: {:.2f}h"\
        .format((time()-exp_start)/3600.)
print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
costs = []
lowest_valid_cost = numpy.finfo(numpy.float32).max
corresponding_test_cost = numpy.finfo(numpy.float32).max
new_lowest_cost = False
end_of_batch = False
epoch = 0

h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM), dtype='float32')
big_h0 = numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*BIG_DIM), dtype='float32')

### load para and generate
#find the para to load
FLAG_FOUND_EXP = False
FLAG_FOUND_PARA = False

results_dir = '/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/results_3t/'
for exp_dir in os.listdir(results_dir):
    if EXP_NAME in exp_dir:
        FLAG_FOUND_EXP = True
        break
        
if FLAG_FOUND_EXP:
    para_dir = results_dir+exp_dir+'/params/'
    for para_name in os.listdir(para_dir):
        if EP_IDX in para_name:
            FLAG_FOUND_PARA = True
            break

#load and generate
if FLAG_FOUND_PARA:
    print('---found para pkl, loading---')
    paraPath = para_dir + para_name
    print paraPath
    lib.load_params(paraPath)
    print('---loading complete---')

    tag = '8s_' + EP_IDX
    print('generating 8 seconds of speech: sampling')
    generate_and_save_samples(tag)
    print('generating 8 seconds of speech: ok')
    
else:
    print('---did not find para pkl, exiting---')

sys.exit()