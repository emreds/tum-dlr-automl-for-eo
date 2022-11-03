import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

LABELS_DICT = \
    {
        0: "Compact high_rise",
        1: "Compact middle_rise",
        2: "Compact low_rise",
        3: "Open high_rise",
        4: "Open middle_rise",
        5: "Open low_rise",
        6: "Lightweight low_rise",
        7: "Large low_rise",
        8: "Sparsely built",
        9: "Heavy industry",
        10: "Dense trees",
        11: "Scattered trees",
        12: "Bush or scrub",
        13: "Low plants",
        14: "Bare rock or paved",
        15: "Bare soil or sand",
        16: "Water",
    }

sentinel2_training_mean = torch.tensor([1.237384229898452759e-01,
                                        1.092514395713806152e-01,
                                        1.010472476482391357e-01,
                                        1.141963005065917969e-01,
                                        1.592123955488204956e-01,
                                        1.814149618148803711e-01,
                                        1.745131611824035645e-01,
                                        1.949533522129058838e-01,
                                        1.542119830846786499e-01,
                                        1.089953780174255371e-01])
sentinel2_training_std = torch.tensor([3.955886885523796082e-02,
                                       4.774657264351844788e-02,
                                       6.632962822914123535e-02,
                                       6.356767565011978149e-02,
                                       7.745764404535293579e-02,
                                       9.104172885417938232e-02,
                                       9.217569977045059204e-02,
                                       1.016793847084045410e-01,
                                       9.986902773380279541e-02,
                                       8.776713907718658447e-02])
sentinel2_validation_mean = torch.tensor([1.289583146572113037e-01,
                                          1.164120137691497803e-01,
                                          1.122049614787101746e-01,
                                          1.240097358822822571e-01,
                                          1.646174490451812744e-01,
                                          1.862829625606536865e-01,
                                          1.791501641273498535e-01,
                                          2.004020363092422485e-01,
                                          1.738285273313522339e-01,
                                          1.277212053537368774e-01
                                          ])
sentinel2_testing_mean = torch.tensor([1.278449594974517822e-01,
                                       1.149842068552970886e-01,
                                       1.111395284533500671e-01,
                                       1.232199594378471375e-01,
                                       1.645713448524475098e-01,
                                       1.862128973007202148e-01,
                                       1.792910993099212646e-01,
                                       2.002600133419036865e-01,
                                       1.727724820375442505e-01,
                                       1.278162151575088501e-01
                                       ])
sentinel2_testing_std = torch.tensor([3.514893725514411926e-02,
                                      4.023178666830062866e-02,
                                      5.523603409528732300e-02,
                                      5.091508477926254272e-02,
                                      6.154564023017883301e-02,
                                      7.297030836343765259e-02,
                                      7.590688019990921021e-02,
                                      8.254054188728332520e-02,
                                      8.815932273864746094e-02,
                                      8.100783824920654297e-02,
                                      ])


class So2SatDataSet(Dataset):
    """
    So2Sat LCZ42 Version2, containing training, validation, testing splits
    https://mediatum.ub.tum.de/1483140
    """

    def __init__(self,
                 file_path=None,
                 sentinel=None,
                 transform=None
                 ):
        """
        :param root_dir: path to the .h5 file with annotations
        :param transform (callable, optional): optional transform to be applied to data
        """
        self.file_path = file_path
        self.sentinel = sentinel
        self.transform = transform

        self.data = h5py.File(self.file_path)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        label = self.data['label'][index]
        label_name = LABELS_DICT[np.where(label == 1.0)[0][0]]

        if self.sentinel == "Sentinel-1":
            image = self.data['sen1'][index]
        elif self.sentinel == "Sentinel-2":
            image = self.data['sen2'][index]

        if self.transform:
            image = self.transform(image)

        return image, label_name
