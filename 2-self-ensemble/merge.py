import json

from utils import *

f1 = get_current_dir() / "self_ensemble-v1.jsonl"
f2 = get_current_dir() / "self_ensemble-v2.jsonl"
output_f = get_current_dir() / "self_ensemble-merged.jsonl"

content1 = open(f1).read()
content2 = open(f2).read()
assert len(content1.split("\n")) == len(content2.split("\n"))

content1 = "\n".join([l for l in content1.split("\n") if l.strip()])
content2 = "\n".join([l for l in content2.split("\n") if l.strip()])

for l1, l2 in zip(content1.split("\n"), content2.split("\n")):
    l1_dict = json.loads(l1)
    l2_dict = json.loads(l2)

    # get key names from l2
    l2_keys = list(l2_dict.keys())

    # replace values in l1 with values in l2
    for k in l2_keys:
        l1_dict[k] = l2_dict[k]

    # turn back to json
    l1_new = json.dumps(l1_dict)
    print(json.dumps(json.loads(l1_new), indent=4))

    # write to file
    with open(output_f, "a") as f:
        f.write(l1_new + "\n")
