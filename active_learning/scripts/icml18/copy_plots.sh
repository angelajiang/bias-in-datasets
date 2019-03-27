PLOT_DIR="/Users/angela/src/private/bias-in-datasets/active_learning/plots/icml18/"
PAPER_DIR="/Users/angela/src/private/papers/selective-backprop-paper/icml2019/figs/"

# CIFAR10 0.1
cp -r $PLOT_DIR"/181208_net/mobilenetv2/Selective-Backprop (Us)_Training Iterations_Test Accuracy_lr0.0.pdf" $PAPER_DIR/accuracy/mobilenetv2-0.1.pdf

cp -r $PLOT_DIR"/181208_net/densenet/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/densenet-0.1.pdf
cp -r $PLOT_DIR"/181208_net/mobilenetv2/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/mobilenetv2-0.1.pdf
cp -r $PLOT_DIR"/181208_net/resnet/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/resnet-0.1.pdf

cp -r $PLOT_DIR"/181208_net/mobilenetv2/Selective-Backprop (Us)_Training Iterations_Test Loss_lr0.0.pdf" $PAPER_DIR/loss/mobilenetv2-0.1.pdf

cp -r $PLOT_DIR"/181208_net/mobilenetv2/Selective-Backprop (Us)_Training Iterations_Ratio Backpropped_lr0.0.pdf" $PAPER_DIR/ratio/mobilenetv2-0.1.pdf

# MNIST 0.0, 0.1
cp -r $PLOT_DIR"181230//Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.001.pdf" $PAPER_DIR/error/mnist-0.0-0.1.pdf

# CIFAR100 0.1
cp -r $PLOT_DIR"icml18_lite//Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/cifar100-0.1.pdf

# SVHN 0.1
cp -r $PLOT_DIR"190118_giulio//Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.01.pdf" $PAPER_DIR/error/svhn-0.01.pdf

# TOPK, CIFAR10, 0.1
#cp -r $PLOT_DIR"181212_topk//mobilenet/TopK_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/topk-mobilenetv2-0.1.pdf
cp -r $PLOT_DIR"190116_dists//Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/probs-mobilenetv2-0.1.pdf

# Label error
cp -r $PLOT_DIR"190116_shuffle//0.001/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/labelerror-mobilenetv2-0.001.pdf
cp -r $PLOT_DIR"190116_shuffle//0.01/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/labelerror-mobilenetv2-0.01.pdf
cp -r $PLOT_DIR"190116_shuffle//0.1/Selective-Backprop (Us)_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/labelerror-mobilenetv2-0.1.pdf

# Target confidences, CIFAR10, 0.1
#cp -r $PLOT_DIR"190108_confidences/target_confidences.pdf" $PAPER_DIR/confidences/mobilenet-0.1.pdf
cp -r $PLOT_DIR"190108_confidences/Target Confidence.pdf" $PAPER_DIR/confidences/confidence-mobilenet-0.1.pdf
cp -r $PLOT_DIR"190108_confidences/Percent Correct.pdf" $PAPER_DIR/confidences/accuracy-mobilenet-0.1.pdf

# Katharopoulos
cp -r $PLOT_DIR"190120_kath/Katharopoulos18-Uniform_Training Iterations_Test Error Percent_lr0.0.pdf" $PAPER_DIR/error/kath.pdf

# Wall clock speedup
cp -r $PLOT_DIR"/speedup/cifar10.pdf" $PAPER_DIR/speedup/cifar10.pdf
