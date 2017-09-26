LOGDIR=./training_log
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./model/VGG_ILSVRC_16_layers.caffemodel
GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

