# sampleRNN_speech


# To run, change --batch_size & --which_set:
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier.py --exp TRY_UTT --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 3 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 1 --weight_norm True --learn_h0 True --which_set SPEECH

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/three_tier/three_tier.py --exp UTT_SQL1024 --seq_len 1024 --big_frame_size 8 --frame_size 2 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 1 --weight_norm True --learn_h0 True --which_set SPEECH
