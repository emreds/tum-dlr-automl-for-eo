from collections import Counter
import json

class CollectTrainedArchs:
    '''
        Given the dictionaries and lists, puts them into proper format for 
        filtering the trained architectures.
    '''
    @staticmethod    
    def load_json(json_path):
        with open(json_path) as f:
            py_json = json.load(f)

        return py_json
    
    def __init__(self, all_samples_path, test_results_path, sequences_path):
        self.all_sampled_archs = self.load_json(all_samples_path)
        self.trained_archs = self.load_json(test_results_path)
        self.sequences = self.load_json(sequences_path)[0]
        self.samples = []
        self.starting_points = set()
        self.code_hash_id = {}
        
    def get_starting_points(self, seq_len=6): 
        """
        Returns the starting points with the `seq_len`.
        """
        #seq_lens = [len(self.sequences[arch]) for arch in self.sequences]
        for arch in self.sequences: 
            if len(self.sequences[arch]) >= seq_len and arch in self.trained_archs: 
                self.samples.append(arch)
                
        self.starting_points = set(self.samples)
        
        return self.starting_points
    
    def get_code_hash_id_map(self, hash_to_id):
        """
        Maps the architecture code into hash and id.
        """
        for arch_code in self.trained_archs: 
            for item in self.all_sampled_archs:
                if item['arch_code'] == arch_code: 
                    self.code_hash_id[arch_code] = {"hash": item["unique_hash"], "id": -1} 
                    break
        
        for code, values in self.code_hash_id.items():
            self.code_hash_id[code]["id"] = hash_to_id[values["hash"]]
        
        return self.code_hash_id
        
    def get_trained_ids(self, hash_to_id):
        """
        Returns the ids of the trained architectures.
        """
        self.get_code_hash_id_map(hash_to_id)
        trained_ids = set([self.code_hash_id[code]["id"] for code in self.code_hash_id])
        
        return trained_ids
    
    def get_starting_ids(self, hash_to_id={}):
        """
        Returns the starting ids.
        """
        self.get_starting_points()
        if not self.code_hash_id:
            assert len(hash_to_id) > 0, "To get the starting id's you should give `hash_to_id` dictionary."
            self.get_code_hash_id_map(hash_to_id)
        starting_ids = sorted([self.code_hash_id[point]["id"] for point in self.starting_points])
        
        return starting_ids
        
    