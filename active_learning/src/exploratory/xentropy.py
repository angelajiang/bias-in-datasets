import ast

def process(input_file):
    with open(input_file) as f:
        for line in f:
            if "xentropy" in line:
                vals = line.split(";")
                softmax = ast.literal_eval(vals[1])
                xentropy = vals[2]
                print(softmax)
                print(xentropy)
                exit()

if __name__ == "__main__":
  input_file = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/loss_fn_exploration/out"
  process(input_file)
