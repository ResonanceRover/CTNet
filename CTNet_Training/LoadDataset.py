import torch
import scipy.io as sio

def load_mat_to_torch(file_path, key_name='TFR'):
    """
    Load a specified key (defaulting to 'TFR') from a ".mat" file and convert it to a PyTorch tensor.

    Parameters:
    file_path (str): The path to the .mat file.
    key_name (str): The key name in the .mat file to load, defaults to 'TFR'.
    Returns:
    torch.Tensor: The loaded and converted PyTorch tensor.
    """

    mat = sio.loadmat(file_path)
    if key_name not in mat:
        raise KeyError(f"Key '{key_name}' not found in .mat file.")

    data = mat[key_name]
    tfr_tensor = torch.from_numpy(data)

    return tfr_tensor
