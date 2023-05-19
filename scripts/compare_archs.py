import json

with open("./sampled_archs_new/arch_specs.json") as json_file:
    new_archs = json.load(json_file)
    
with open("../../../training/sampled_archs/arch_specs.json") as json_file:
    old_archs = json.load(json_file)

not_in_olds_cnt = 0

print(f"Len of old_archs {len(old_archs)}")

old_binaries = {arch["binary_encoded"] for arch in old_archs}

for arch in new_archs: 
    if arch["binary_encoded"] not in old_binaries:
        not_in_olds_cnt += 1
        
print(f"These archs are not in old binaries: {not_in_olds_cnt}")
