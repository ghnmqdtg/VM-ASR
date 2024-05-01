from torch import optim
import itertools


def get_optimizer(config, models, logger, **kwargs):
    """
    Build optimizer, setting weight decay of normalization to 0 by default for multiple models.
    """
    logger.info(
        f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}...................."
    )

    # Convert single model to list if not already a list
    if not isinstance(models, (list, tuple)):
        models = [models]

    skip_list = []
    skip_keywords = []
    for model in models:
        if hasattr(model, "no_weight_decay"):
            skip_list.extend(model.no_weight_decay())
        if hasattr(model, "no_weight_decay_keywords"):
            skip_keywords.extend(model.no_weight_decay_keywords())

    # Get all parameters from all models
    all_parameters, no_decay_names = set_weight_decay(models, skip_list, skip_keywords)
    logger.info(f"No weight decay list: {no_decay_names}")

    # Select optimizer
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            all_parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            all_parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError

    return optimizer


def set_weight_decay(models, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    no_decay_names = []

    # Iterate over all models
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen weights
            if (
                len(param.shape) == 1
                or name.endswith(".bias")
                or name in skip_list
                or check_keywords_in_name(name, skip_keywords)
            ):
                no_decay.append(param)
                no_decay_names.append(name)
            else:
                has_decay.append(param)

    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], no_decay_names


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)
