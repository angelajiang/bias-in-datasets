#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/flowers_train /tmp/flowers_train_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/imagenet_train/validation/ /tmp/imagenet_validation_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/cifar10_resnet /tmp/cifar10_resnet_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/iii-buses-noloss_inception/ /tmp/bus_inception_narwhal

#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/flowers_train/events.out.tfevents.1535641129.h0.bias.biglearning.narwhal.pdl.cmu.edu /tmp/flowers_train_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/imagenet_train/validation/events.out.tfevents.1535641047.h0.bias-2.biglearning.narwhal.pdl.cmu.edu /tmp/imagenet_validation_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/cifar10_resnet/events.out.tfevents.1535640819.h1.bias-2.biglearning.narwhal.pdl.cmu.edu /tmp/cifar10_resnet_narwhal
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/iii-buses-noloss_inception/events.out.tfevents.1535640767.h0.bias-4.biglearning.narwhal.pdl.cmu.edu /tmp/bus_inception_narwhal

#EXPERIMENT_NAME=180703
#OUTPUT_DIR=/tmp/$EXPERIMENT_NAME
#mkdir $OUTPUT_DIR
#
#FLOWERS_OUTPUT=$OUTPUT_DIR/flowers
#IMAGENET_OUTPUT=$OUTPUT_DIR/imagenetval
#CIFAR10_OUTPUT=$OUTPUT_DIR/cifar10
#BUSES_OUTPUT=$OUTPUT_DIR/buses
#mkdir $FLOWERS_OUTPUT
#mkdir $IMAGENET_OUTPUT
#mkdir $CIFAR10_OUTPUT
#mkdir $BUSES_OUTPUT
#
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/bus_inception/event* $BUSES_OUTPUT
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/cifar10_inception/event* $CIFAR10_OUTPUT
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/flowers_inception/event* $FLOWERS_OUTPUT
#scp -r ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/imagenetval_inception/event* $IMAGENET_OUTPUT
#
EXPERIMENT_NAME=180904
OUTPUT_DIR=/tmp/$EXPERIMENT_NAME
mkdir $OUTPUT_DIR

BUS_0=$OUTPUT_DIR/buses_0
BUS_3=$OUTPUT_DIR/buses_3
BUS_4=$OUTPUT_DIR/buses_4
BUS_10=$OUTPUT_DIR/buses_10
mkdir $BUS_0
mkdir $BUS_3
mkdir $BUS_4
mkdir $BUS_10

scp ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/bus_inception_0/event* $BUS_0
scp ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/bus_inception_3/event* $BUS_3
scp ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/bus_inception_4/event* $BUS_4
scp ahjiang@ops.narwhal.pdl.cmu.edu:/proj/BigLearning/ahjiang/output/tmp/$EXPERIMENT_NAME/bus_inception_10/event* $BUS_10
