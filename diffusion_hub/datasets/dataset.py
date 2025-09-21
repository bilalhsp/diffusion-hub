
from torchvision import datasets, transforms
import warnings


__DATASET__ = {}

def register_dataset(name):
    def wrapper(cls):   
        if __DATASET__.get(name, None):
            if __DATASET__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DATASET__[name] = cls
        cls.name = name
        return cls
    return wrapper

def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)


@register_dataset("mnist")
class MNISTDataset:
    """
    General MNIST dataset wrapper.
    Can be instantiated for train or test split.
    Implements __len__ and __getitem__ so it behaves like a standard dataset.
    """
    def __init__(self, data_dir="./data", train=True, download=True):
        self.root = data_dir
        self.train = train
        self.download = download

        # Transform: convert to tensor and scale to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Pad(2),                 # pads 2 pixels on all sides: 28+2*2 = 32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Initialize underlying torchvision MNIST dataset
        self.dataset = datasets.MNIST(
            root=self.root, train=self.train, transform=self.transform, download=self.download
        )

    def __len__(self):
        # Forward to MNIST dataset's len
        return len(self.dataset)

    def __getitem__(self, index):
        # Forward to MNIST dataset's getitem
        return self.dataset[index]
