from collections import Counter

class CollectTrainedArchs:
    def __init__(self, all_samples_path, test_results_path, sequences_path):
        self.all_sampled_archs = load_json(all_samples_path)
        self.trained_archs = load_json(test_results_path)
        self.sequences = load_json(sequences_path)
        self.samples = []
        self.starting_points = set()
        self.code_hash_id = {}
        
    def get_starting_points(self, seq_limit=6): 
        seq_lens = [len(self.sequences[arch]) for arch in self.sequences]
        for arch in sequences: 
            if len(sequences[arch]) >= seq_limit and arch in trained_archs: 
                self.samples.append(arch)
                
        self.starting_points = set(self.samples)
        
        return starting_points
    
    def get_code_hash_id_map(self, hash_to_id):
        for arch_code in self.trained_archs: 
            for item in self.all_existing_archs:
                if item['arch_code'] == arch_code: 
                    self.code_hash_id[arch_code] = {"hash": item["unique_hash"], "id": -1} 
                    break
        
        for code, values in code_hash_id.items(): 
            self.code_hash_id[code]["id"] = hash_to_id[values["hash"]]
        
        return self.code_hash_id
        
    def get_trained_ids(self, hash_to_id):
        get_code_hash_id_map(hash_to_id)
        trained_ids = set([self.code_hash_id[code]["id"] for code in self.code_hash_id])
        
        return trained_ids
    
    def get_starting_ids(self, hash_to_id={}):
        get_starting_points()
        if not self.code_hash_id:
            assert(hash_to_id, "To get the starting id's you should give `hash_to_id` dictionary.")
            get_code_hash_id_map(hash_to_id)
        starting_ids = sorted([self.code_hash_id[point]["id"] for point in self.starting_points])
        set_starting_ids = set(starting_ids)
        
        return starting_ids
        
    @staticmethod    
    def load_json(json_path):
    with open(json_path) as f:
        py_json = json.load(f)

    return py_json