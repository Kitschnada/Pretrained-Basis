import os
import torch
import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset
from .splits import saved_split_indices
from .vision_downloads import prepare_vision_task


ROOT = os.path.join(os.getenv("PGSB_VISION_DATA_ROOT", "data"), "stanford_cars")


class PytorchStanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str = ROOT,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        # Keep the legacy aliases but use one canonical value in all branches.
        split = {"cars_train": "train", "cars_test": "test"}.get(split, split)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root)
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        # The original Stanford archive URLs are gone. Preserve original JPEG
        # filenames and MAT annotation order through a pinned raw-file mirror.
        prepare_vision_task("stanford_cars", self._base_folder.parent,
                            cars_splits=(self._split,), cars_root=self._base_folder)

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit" / "cars_meta.mat").is_file():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


class Cars:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        if is_train:
            self.train_dataset = PytorchStanfordCars(location, 'cars_train', preprocess, download=True)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            dataset = self.train_dataset
        else:
            self.test_dataset = PytorchStanfordCars(location, 'cars_test', preprocess, download=True)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
            dataset = self.test_dataset

        idx_to_class = dict((v, k)
                            for k, v in dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]


def prepare_train_loaders(config):

    dataset_class = Cars(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders


def prepare_test_loaders(config):
    dataset_class = Cars(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    loaders = {
        'test': dataset_class.test_loader
    }

    if config.get('val_fraction', 0) > 0.:
        print('splitting cars')
        test_set = loaders['test'].dataset
        valid_idxs, test_idxs = saved_split_indices(test_set, config)

        val_set = torch.utils.data.Subset(test_set, valid_idxs)
        test_set = torch.utils.data.Subset(test_set, test_idxs)

        loaders['test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        loaders['val'] = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
    loaders['class_names'] = dataset_class.classnames

    return loaders
