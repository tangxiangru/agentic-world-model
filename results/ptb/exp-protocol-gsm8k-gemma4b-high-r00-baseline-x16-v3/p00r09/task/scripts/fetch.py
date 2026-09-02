import datasets, json, sys
which = sys.argv[1]
if which == "omi":
    d = datasets.load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    print(d)
    print({k: d[0][k] for k in d.column_names})
elif which == "meta":
    d = datasets.load_dataset("meta-math/MetaMathQA", split="train")
    print(d)
    print(d[0])
