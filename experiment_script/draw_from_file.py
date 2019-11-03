import matplotlib.pyplot as plt
import numpy
import re
import argparse
import json

'''
json file format:
{
    logs:
    [
        {
            "exp_name": xxx,
            "path": xxx
        },
        ...
    ],
    targets:
    [
        {
            "name": xxx,
            "re": xxx,
            "output_path": xxx
        },
        ...
    ]
}
'''

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_log(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def extract_stats(log_content, targets):
    result_dict = {}
    for t in targets:
        result_dict[t["name"]] = list(map(lambda x:float(x), re.findall(t["re"], log_content)))
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="path to plot configuration file")
    args = parser.parse_args()
    config = load_config(args.config)

    logs = config["logs"]
    targets = config["targets"]

    result_dict = {}
    for l in logs:
        result_dict[l["exp_name"]] = extract_stats(load_log(l["path"]), targets)
    
    for t in targets:
        name = t["name"]
        plt.figure()
        plt.ylabel(name)
        plt.xlabel("epoch")
        plt.grid(True)
        for exp_name, res_dict in result_dict.items():
            plt.plot(range(1, len(res_dict[name])+1), res_dict[name], label=exp_name)
        plt.legend()
        plt.show()
        # plt.savefig(t["output_path"])
        plt.close()
