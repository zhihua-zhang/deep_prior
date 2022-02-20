import torch

import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

###############################################################################
# (a) Load data & Preprocess
###############################################################################

def selectData(ds, args):
    n_sp = args.n_sp
    data_unsp_all = []
    data_sp_all = []
    for digit in range(10):
      loc = ds.targets == digit
      data, target = map(lambda x: x[loc], [ds.data, ds.targets])

      data_sp_all.append([data[:n_sp], target[:n_sp]])
      data_unsp_all.append([data[n_sp:], target[n_sp:]])

    data_sp, target_sp = map(torch.cat, zip(*data_sp_all))
    data_unsp, target_unsp = map(torch.cat, zip(*data_unsp_all))

    if args.base_model == "fc":
      data_sp, data_unsp = map(lambda x: x.reshape(len(x), -1).float(), [data_sp, data_unsp])
    else:
      data_sp, data_unsp = map(lambda x: x.unsqueeze(1).float(), [data_sp, data_unsp])
    target_sp, target_unsp = map(lambda x: x.long(), [target_sp, target_unsp])
      
    return data_sp, target_sp, data_unsp, target_unsp

def normalize(data):
    data = data / 255.
    # data = (data - 0.5) / 0.5
    return data

def preprocess(data):
    data = normalize(data)
    return data
  
def random_shuffle(data, target, seed=2022):
    idx = torch.randperm(len(data))
    data, target = map(lambda x: x[idx], [data, target])
    return data, target

class myDataset(torch.utils.data.Dataset):
  def __init__(self, images, labels, transform=None):
    super().__init__()
    
    self.images = images
    self.labels = labels
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels[idx]
    return image, label

def get_dataset(args):
  train_ds = torchvision.datasets.MNIST(root="../data/training.pt",
                                   train=True,
                                   download=True,
                                   transform=T.Compose([
                                       T.PILToTensor(),
                                   ]))

  val_ds = torchvision.datasets.MNIST(root="../data/validation.pt",
                                    train=False,
                                    download=True,
                                    transform=T.Compose([
                                        T.PILToTensor()
                                    ]))  
  ### Train
  # sp /unsp dataset
  data_sp, target_sp, data_unsp, target_unsp = selectData(train_ds, args)

  ### Val
  if args.base_model == "fc":
    data_val = val_ds.data.reshape(len(val_ds.data), -1).float()
  else:
    data_val = val_ds.data.unsqueeze(1).float()
  
  target_val = val_ds.targets.long()

  # pre-process (normalize to [0,1])
  data_sp, data_unsp, data_val = map(preprocess, [data_sp, data_unsp, data_val])
  
  # random shuffle
  data_unsp, target_unsp = random_shuffle(data_unsp, target_unsp)
  data_sp, target_sp = random_shuffle(data_sp, target_sp)
  
  train_ds_unsp = myDataset(data_unsp, target_unsp)
  train_ds_sp = myDataset(data_sp, target_sp)
  val_ds = myDataset(data_val, target_val)
  return train_ds_unsp, train_ds_sp, val_ds

def get_loader(args, seed=2022):
  torch.manual_seed(seed)
  
  n_order = args.n_order
  bz_sp = args.bz_sp
  bz_unsp = 7 * bz_sp
  assert bz_unsp % n_order == 0, f"bz_unsp:{bz_unsp}, n_order:{n_order} - unsupervised batch size is not a multiple of n_order"
  
  train_ds_unsp, train_ds_sp, val_ds = get_dataset(args)

  train_loader_unsp = torch.utils.data.DataLoader(train_ds_unsp, batch_size=bz_unsp, shuffle=True)
  train_loader_sp = torch.utils.data.DataLoader(train_ds_sp, batch_size=bz_sp, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

  print("train dataset(unsupervise) n_sample:{n_sample}, shape:{shape}".format(n_sample=len(train_ds_unsp), shape=train_ds_unsp[0][0].shape))
  print("train dataset(supervise)   n_sample:{n_sample}, shape:{shape}".format(n_sample=len(train_ds_sp), shape=train_ds_sp[0][0].shape))
  print("val   dataset n_sample:{n_sample}, shape:{shape}".format(n_sample=len(val_ds), shape=val_ds[0][0].shape))
  print("train loader(unsupervise) bz:{bz}".format(bz=bz_unsp))
  print("train loader(supervise)   bz:{bz}".format(bz=bz_sp))
  print("val   loader bz:{bz}".format(bz=bz_sp))
  return train_loader_unsp, train_loader_sp, val_loader


# if __name__ == "__main__":
#     from main import get_args
#     args = get_args()
    
#     train_ds_unsp, train_ds_sp, val_ds = get_dataset(args)
#     train_loader_unsp, train_loader_sp, val_loader = get_loader(args)