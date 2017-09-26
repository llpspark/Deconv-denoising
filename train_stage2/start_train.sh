LOGDIR=./training_log
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=/home/gt/llp-Deconv1/train_stage1/im2noi_vgg_decon_newresi/result/im2noi_stage_1_iter_200000.caffemodel
GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

