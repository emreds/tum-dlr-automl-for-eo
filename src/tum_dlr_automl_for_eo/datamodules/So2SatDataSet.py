import h5py
import numpy as np
from torch.utils.data import Dataset

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

        :param file_path: path to read data
        :param sentinel:  Sentinel-1 or Sentinel-2
        :param transform: torchvision.transforms, transformation to be applied
                          data
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
        # target is one-hot encoded, convert to scalar
        label = np.where(label == 1.0)[0][0]

        if self.sentinel == "Sentinel-1":
            image = self.data['sen1'][index]
        elif self.sentinel == "Sentinel-2":
            image = self.data['sen2'][index]

        if self.transform:
            image = self.transform(image)

        return image, label
