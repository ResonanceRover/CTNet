import torch

def normalize_batch(batch):
    """
    Normalize a batch of matrices by subtracting the minimum value and dividing by the range.
    """
    # print('batch.shape', batch.shape)
    min_value = torch.min(batch)
    # print('min_val', min_value)
    max_value = torch.max(batch)
    # print('max_value', max_value)

    # To check if the minimum and maximum values are equal in order to avoid division by zero.
    if max_value == min_value:
        return torch.zeros_like(batch)
    else:
        return (batch - min_value) / (max_value - min_value)
def compute_batch_nl1d(Y_batch, gt_batch):
    """
    Compute the NL1D for a batch of samples Y_batch and ground truth gt_batch.

    Y_batch and gt_batch should have shape (batch_size, Nf, L).
    Nf=200, L=400 in this case.
    """
    batch_size, Nf, L = Y_batch.shape
    assert Nf == 200 and L == 400, "Expected Nf=200 and L=400"

    Y_norm = normalize_batch(Y_batch)
    # print('Y_norm',Y_norm.shape)
    gt_norm = normalize_batch(gt_batch)

    # Compute L1 distance for each sample (axis=0 represents the batch dimension)
    l1_distances = torch.sum(torch.abs(Y_norm - gt_norm), dim=(1, 2))

    # Optionally, normalize by the total number of elements (Nf * L) if you want a true NL1D
    nl1d_batch = l1_distances / (2*Nf * L)  # Uncomment this line if you want NL1D normalized by Nf * L
    nl1d_batch_mean = torch.mean(nl1d_batch)
    # Return the L1 distances for each sample in the batch
    return nl1d_batch_mean


def l2_distance(y_true, y_pred):
    """
    Calculate l2 distance

    Parameters:
    y_true -- An array of predicted values
    y_pred -- An array of true values
    """
    # Ensure that the inputs are NumPy arrays for convenient subsequent calculations
    loss_l2 = torch.pow((y_true -y_pred), 2)
    # print('loss_l2',loss_l2.shape)
    loss_sum = torch.sum(loss_l2).to(torch.float32)
    # print('loss_fr', loss_fr)

    # Calculate mse value
    # mse = np.mean((y_true - y_pred) ** 2)
    # print('mse', mse.shape)
    # mse =mse.to(torch.float32)
    return loss_sum


