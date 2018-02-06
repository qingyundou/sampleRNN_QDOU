#0 preparation
NAME=UC_UNRP_RS #////////////////////////////////////TBC
#ulaw, normed already, remove zero, precise lab (or not) when running conditional model
#remove DC offset, fix bug in rmzero by copying silence



#1 for running on grid: copy data to air
cd /home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/scripts
python check_move_data.py




#2 main code, can be run without considering data transfer

#2.1 for running on grid: go to right directory
cd /home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/

#2.2 set up env
source activate theano
export PATH=${PATH}:/usr/local/cuda/bin
export THEANO_FLAGS="mode=FAST_RUN,device=gpu$X_SGE_CUDA_DEVICE,floatX=float32"
unset LD_PRELOAD

#2.3 key cmd
#---------------------------key cmd
python -u models/three_tier/three_tier.py --exp ${NAME} --seq_len 800 --big_frame_size 80 --frame_size 10 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 3 --rnn_type GRU --q_levels 256 --q_type mu-law --batch_size 20 --weight_norm True --learn_h0 True --which_set SPEECH --n_big_rnn 1 --normed --rmzero --grid --resume
#---------------------------key cmd

#2.4 move log file
mv run_${NAME}.sh.* scripts/autolog/




#3 for running on grid: remove data from air (optional)
cd /home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/scripts
python check_remove_data.py
