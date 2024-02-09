from functools import partial
from itertools import islice

import jax.numpy as jnp
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from tqdm import tqdm

def trfm(size, img, channels=1):
    t = TF.to_tensor(img)
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    numpy_img = t.squeeze(0).permute(1,2,0).numpy()
    out = jnp.array(numpy_img)
    if out.shape[-1] != channels:
        if channels == 1:
            out = out.mean(axis=-1, keepdims=True)
        else:
            assert out.shape[-1] == 1, \
                f"incompatible channel number: expected {channels} but got {out.shape[-1]}"
            out = jnp.tile(out, (3,))
    return out

def smallnist(N, classes, size, train=True, root="../datasets"):
    dataset = iter(MNIST(root, train=train, download=True, transform=partial(trfm, size)))
    imgs = []
    labels = []
    while len(imgs) < N:
        img, label = next(dataset)
        if label in classes:
            labels.append(classes.index(label))
            imgs.append(img)
        else:
            continue
    return jnp.array(imgs), jnp.array(labels)

def smallfnist(N, classes, size, train=True, root="../datasets"):
    dataset = iter(FashionMNIST(root, train=train, download=True, transform=partial(trfm, size)))
    imgs = []
    labels = []
    while len(imgs) < N:
        img, label = next(dataset)
        if label in classes:
            labels.append(classes.index(label))
            imgs.append(img)
            pbar.update(1)
        else:
            continue
    return jnp.array(imgs), jnp.array(labels)

def get_dataset(root, name, train, resolution):
    if name == "mnist":
        return MNIST(root, train=train, download=True, transform=partial(trfm, resolution))
    elif name == "fmnist":
        return FashionMNIST(root, train=train, download=True, transform=partial(trfm, resolution))
    elif name == "cifar10":
        return CIFAR10(root, train=train, download=True, transform=partial(trfm, resolution))
    else:
        raise NotImplementedError(f"Dataset '{name}' not recognised.")

def get_chunk(dataset, labels, remap, N, start=0):
    elems = tqdm(((d, remap(l)) for d, l in iter(dataset) if l in labels), total=N, desc="Compiling data tranch")
    imgs, labels = map(lambda gen: jnp.array(list(gen)), zip(*islice(elems, start, N)))
    return imgs, labels

def cfg_tranch(defaults, tranch, resolution):
    def get(key):
        return tranch[key] if key in tranch else defaults[key].get()
    N = get('N')
    TN = get('TN')
    classes = get('classes')

    dataset = get('dataset')
    root = get('root')
    remap_val = get('remap')
    remap = lambda x: x if remap_val is None else remap_val[classes.index(x)]

    train = get_chunk(get_dataset(root, dataset, True, resolution), classes, remap, len(classes)*N)
    test = get_chunk(get_dataset(root, dataset, False, resolution), classes, remap, len(classes)*TN)

    return train, test

def cfg_tranches(cfg, resolution):
    defaults = cfg['defaults']
    return list(cfg_tranch(defaults, tranch, resolution) for tranch in cfg['tranches'].get())

def tranch_cat(tranches, index, train):
    tups = list(islice([tranch[0] if train else tranch[1] for tranch in tranches], index+1))
    return tuple([jnp.concatenate(arrs, axis=0) for arrs in zip(*tups)])
