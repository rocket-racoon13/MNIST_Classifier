from typing import Any, Tuple, Optional, Callable
import os
import gzip
import numpy as np


class VisionDataset:
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = root
        self.data = None
        self.targets = None
        
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")
        
        self.transform = transform
        self.target_transform = target_transform
        
        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError


class EvalDataset:
    def __init__(
        self,
        args,
        root: str,
        transform: Optional[Callable] = None
    ):
        self.args = args
        self.data_list = [os.path.join(root, image_file_name) for image_file_name in os.listdir(root)]
        self.transform = transform
        
    def __getitem__(self, idx: int) -> Any:
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(self.args, data)
        return data
    
    def __len__(self) -> int:
        return len(self.data_list)


class MNIST(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(root, transform, transforms, target_transform)
        self.train = train
        
        if self.train:
            image_file_name = "train-images-idx3-ubyte.gz"
            label_file_name = "train-labels-idx1-ubyte.gz"
        else:
            image_file_name = "t10k-images-idx3-ubyte.gz"
            label_file_name = "t10k-labels-idx1-ubyte.gz"
            
        self.data = self.load_images(image_file_name)   # Unzip MNIST image and convert to np.ndarray
        self.targets = self.load_labels(label_file_name)   # Unzip MNIST label and convert to np.ndarray
        
        if self.transforms:
            self.data, self.targets = self.transforms(self.data, self.targets)
        
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx], self.targets[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
    # Unzip MNIST image and convert to np.ndarray
    def load_images(self, image_file_name):
        with gzip.open(os.path.join(self.root, image_file_name), 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            image_count = int.from_bytes(f.read(4), 'big')
            row_count = int.from_bytes(f.read(4), 'big')
            column_count = int.from_bytes(f.read(4), 'big')
            image_data = f.read()
            # binary values to np.ndarray
            images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, 1, row_count, column_count))   # binary values to np.ndarray, shape = (60000,1,28,28)
        return images
    
    # Unzip MNIST label and convert to np.ndarray
    def load_labels(self, label_file_name):
        with gzip.open(os.path.join(self.root, label_file_name), 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            label_count = int.from_bytes(f.read(4), 'big')
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)   # binary values to np.ndarray, shape = (10000,)
            return labels
    
    
class StandardTransform:
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.transform = transform
        self.target_transform = target_transform
        
    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target
    
    
def collate_fn(samples):
    pass