import os
import torch
import torchvision.datasets as datasets
from .vision_downloads import prepare_vision_task
from .splits import validate_disjoint_imagefolders


ROOT = os.path.join(os.getenv("PGSB_VISION_DATA_ROOT", "data"), "dtd")


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        prepare_vision_task("dtd", os.path.dirname(location))
        traindir = os.path.join(location, 'train')
        valdir = os.path.join(location, 'val')

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        validate_disjoint_imagefolders(self.train_dataset, self.test_dataset)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]


def prepare_train_loaders(config):
    prepare_vision_task("dtd", os.path.dirname(ROOT))
    train_dataset = datasets.ImageFolder(
        os.path.join(ROOT, "train"),
        transform=config['train_preprocess']
    )

    loaders = {
        'full': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    }

    return loaders


def prepare_test_loaders(config):
    prepare_vision_task("dtd", os.path.dirname(ROOT))

    val_dataset = datasets.ImageFolder(
        os.path.join(ROOT, "val"),
        transform=config['eval_preprocess']
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(ROOT, "test"),
        transform=config['eval_preprocess']
    )

    validate_disjoint_imagefolders(val_dataset, test_dataset)

    idx_to_class = dict((v, k)
                        for k, v in val_dataset.class_to_idx.items())
    classnames = [idx_to_class[i].replace(
        '_', ' ') for i in range(len(idx_to_class))]

    loaders = {
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        ),
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        ),
        'class_names': classnames
    }

    return loaders
