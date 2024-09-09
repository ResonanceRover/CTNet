import os
import torch
import scipy.io as sio
from scipy.io import loadmat
import CTNet_util as util

class Test:
    def __init__(self, data_path):
        self.data_path = data_path
        skip_path = os.path.join('model', 'CTNet_epoch_xxx.pth')
        device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
        try:
            self.skip_module, _, _, _, _ = util.load(skip_path, 'tfa_module', device)
            self.skip_module.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.skip_module = None

    def load_data(self, file_path):
        try:
            mat = loadmat(file_path)
            test_tensor = mat['TFR']
            return test_tensor
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None

    def inference(self, test_tensor):
        if self.skip_module is None:
            print("Model not loaded, cannot perform inference.")
            return

        if len(test_tensor.shape) > 1 and test_tensor.shape[0] >= 3:
            input_tensor = test_tensor[2]
            # inf_tfr = self.skip_module(torch.tensor(input_tensor))
            # Ensure consistency of data type and device
            # input_tensor = torch.from_numpy(input_tensor)#to(self.skip_module.device).float()
            # print('Input shape:', input_tensor.shape)

            with torch.no_grad():
                inf_tfr = self.skip_module(torch.tensor(input_tensor))
                print('Inference shape:', inf_tfr.shape)
                infer_tfr = inf_tfr.cpu().numpy()

                # Save result
            path = os.path.join(self.data_path, 'CTNet_result.mat')
            sio.savemat(path, {'TFR': infer_tfr})


if __name__ == '__main__':
    data_path = 'CTNet/CTNet_Testing'
    test_obj = Test(data_path)
    test_tensor = test_obj.load_data('CTNet/Dataset/xxx.mat')
    if test_tensor is not None:
        test_obj.inference(test_tensor)