import os
import sys

def convert_sbbps_to_bps(filename, target_sbbps):
    with open(filename) as f:
        last_num_backprops = 0
        batch_sizes = []
        for line in f:
            if "test_debug" in line:
                vals = line.split(",")
                epoch = int(vals[1])
                bp = int(vals[2])
                sbbp = int(vals[7])
                if sbbp > target_sbbps and epoch % 10 == 0:
                    return epoch, bp
    return None, 0

if __name__ == "__main__":
    target_sbbps = int(sys.argv[1])
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    dynbatch = "dynbatch_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    reweight_dynbatch = "reweight-dynbatch_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    exps = [
            ("190424_sblr", reweight_dynbatch, "Reweighting + Dynamic Batching + SBLR"),
            #("190424_sblr", dynbatch, "Dynamic Batching + SBLR"),
            ]
    for exp, f, label in exps:
        filename = os.path.join(home, exp, f)
        epoch, bps = convert_sbbps_to_bps(filename, target_sbbps)
        print "{}: epoch{}_{}".format(label, epoch, bps)


