# Match the id with hash code.
# Using hash code match the architecture with the Module adjacency and Module operations.
# Create a dictionary for architecture sequences and their corresponding id, hash_code, module_adjacency, module_operations.
# Write the dictionary to a json file.
# Read that json file and submit those architectures as training job to slurm.
# Make inference and testing on those trained architectures.
# Log all those results.

def arch_logger(sequences, id_to_hash, nb101_dict):
    
    sequences_hash = []
    
    for sequence in sequences:
        sequences_hash.append([(id_to_hash[idx], idx) for idx in sequence])
        
    logs = []
    for sequence in sequences_hash:
        sub_logs = []
        for step, (hash_code, idx) in enumerate(sequence):
            for binary_encode, arch_specs in nb101_dict.items():
                if arch_specs["hash_code"] == hash_code:
                    sub_logs.append({"id": idx, "hash_code": hash_code, "step": step, "module_adjacency": arch_specs["module_adjacency"], "module_operations": arch_specs["module_operations"]})
                    break
        logs.append(sub_logs)
        
    return logs        
    