import torch
import CTNet_model


def load(fn, module_type, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']

    # Adjust args based on device
    args.device = device

    # Ensure module_type is recognized
    if module_type == 'tfa_module':
        model = CTNet_model.tfa_module(args).to(device)
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

    lr = args.lr if module_type != 'ctn' else (args.lr_fr if module_type == 'fr' else args.lr_fc)
    optimizer = optimizer_cls(module.parameters(), lr=lr,
                              alpha=0.9 if module_type != 'ctn' else None)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    return optimizer, scheduler