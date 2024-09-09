import os
import torch
import errno
import TestCTNet


def symlink_force(target, link_name):
    """
    Create a symbolic link named `link_name` pointing to `target`.
    If the link already exists, it will be removed and recreated.

    Parameters:
    - target: The path to the target of the symbolic link.
    - link_name: The name of the symbolic link to create.
    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            try:
                os.remove(link_name)
                os.symlink(target, link_name)
            except OSError as remove_error:

                raise OSError(
                    f"Failed to remove existing link {link_name} before creating a new one: {remove_error}") from e
        else:

            raise


def save_checkpoint(model, optimizer, scheduler, args, epoch, module_type):
    """
    Saves the model, optimizer, scheduler, and epoch to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler to save.
        args (argparse.Namespace): The arguments namespace containing configuration options.
        epoch (int): The current epoch number.
        module_type (str): The type of module being saved (e.g., "CTNet").
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,  # Optional for clarity
        'args': args,
    }

    # Ensure the output directory exists
    output_dir = os.path.join(args.output_dir, module_type)
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True prevents errors if the directory already exists

    # Define the file paths
    last_checkpoint = os.path.join(output_dir, 'last.pth')
    current_checkpoint = os.path.join(output_dir, 'CTNet_epoch_{}.pth'.format(epoch))

    try:
        # Save the checkpoint
        torch.save(checkpoint, current_checkpoint)
        # Optionally, create or update a symbolic link to the latest checkpoint
        symlink_force(current_checkpoint, last_checkpoint)
        print(f"Checkpoint saved to {current_checkpoint}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


def load(fn, module_type, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']

    # Adjust args based on device
    args.device = device

    # Ensure module_type is recognized
    if module_type == 'tfa_module':
        model = TestCTNet.tfa_module(args).to(device)
    else:
        raise ValueError(f'Module type "{module_type}" not recognized')

    model.load_state_dict(checkpoint['model'])

    # Initialize optimizer and scheduler based on args and model
    optimizer, scheduler = optim(args, model, module_type)

    # Load optimizer and scheduler states
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Return loaded model, optimizer, scheduler, args, and epoch
    return model, optimizer, scheduler, args, checkpoint['epoch']

def optim(args, module, module_type):
    optimizer_cls = {
        'ctn': torch.optim.Adam,
        'tfa_module': torch.optim.RMSprop,
    }.get(module_type, None)

    if optimizer_cls is None:
        raise ValueError(f'Expected module_type to be one of {list(optimizer_cls.keys())} but got {module_type}')

    lr = args.lr if module_type != 'ctn' else (args.lr_fr if module_type == 'ctn' else args.lr_fc)
    optimizer = optimizer_cls(module.parameters(), lr=lr,
                              alpha=0.9 if module_type != 'ctn' else None)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    return optimizer, scheduler


def print_args(logger, args):
    """
    Prints and logs command line arguments to the logger and a file.

    Args:
    - logger: A logger object used to record information.
    - args: An object containing all the command line arguments.
    """
    message = ''
    # Sort the attributes of args to ensure a predictable output order
    for k, v in sorted(vars(args).items()):
        # Format the key and value to take up approximately 30 characters each
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
        # Use the logger's info method to record the formatted message
    logger.info(message)

    # Construct the path for the arguments file
    args_path = os.path.join(args.output_dir, 'run.args')
    # Open the file with a 'with' statement to ensure proper closing
    with open(args_path, 'wt') as args_file:
        # Write the formatted message to the file
        args_file.write(message)
        # Add a newline character at the end of the file for cleanliness
        args_file.write('\n')

def parameter_calculation(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params
