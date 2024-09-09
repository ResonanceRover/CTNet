import os
import torch
import scipy.io as sio
from scipy.io import loadmat
import CTNet_util as util

class Test:
    def __init__(self,data_path):
        skip_path = os.path.join('model', 'CTNet_epoch_xxx.pth')
        device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
        self.skip_module, _, _, _, _ = util.load(skip_path, 'tfa_module',device)
        self.skip_module.cpu()
        self.skip_module.eval()
        self.data_path=data_path
    def inference(self):

        input_tensor = test_tensor
        print('a',input_tensor.shape)
        with torch.no_grad():
            inf_tfr = self.skip_module(torch.tensor(input_tensor))
            print('inf_tfr',inf_tfr.shape)
            inf_tfr = inf_tfr.cpu().data.numpy()
            path = os.path.join(self.data_path, 'CTNet_result.mat')
            sio.savemat(path, {'TFR': inf_tfr})



mat = loadmat('CTNet/Dataset/xxx.mat')
test_tensor = mat['TFR']







