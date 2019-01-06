# pytorch-utilities
Data loader and other utility files for PyTorch

## Datasets
All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented. Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples parallelly using torch.multiprocessing workers. All the datasets have almost similar API. They all have a common arguments: transform to transform the input.
