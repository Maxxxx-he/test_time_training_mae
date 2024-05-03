import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Any


class MyPcamDarasets(Dataset):
    def __init__(self, pcam, batch_size: int = 1, steps_per_example: int = 1, minimizer=None,
                 transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        self.pcam = pcam
        self.batch_size = batch_size
        self.minimizer = minimizer
        self.steps_per_example = steps_per_example
        self.single_crop = single_crop
        self.start_index = start_index
        self.transform = transform

    def __len__(self):
        mult = self.steps_per_example * self.batch_size
        mult *= (self.pcam.__len__() if self.minimizer is None else len(self.minimizer))
        return mult

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        real_index = (index // self.steps_per_example) + self.start_index
        if self.minimizer is not None:
            real_index = self.minimizer[real_index]
        sample, target = self.__getitem__(real_index)
        #sample = self.loader(path)
        # if self.transform is not None and not self.single_crop:
        #     samples = torch.stack([self.transform(sample) for i in range(self.batch_size)], axis=0)
        # elif self.transform and self.single_crop:
        #     s = self.transform(sample)
        #     samples = torch.stack([s for i in range(self.batch_size)], axis=0)
        # # if self.target_transform is not None:
        # #     target = self.target_transform(target)

        return sample, target
