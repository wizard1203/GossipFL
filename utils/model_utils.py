from collections.abc import Iterable

import logging




def get_actual_layer_names(model, layer_alias):
    layer_names = []
    name_alias_map = {}
    for alias in layer_alias:
        layer_name = model.layers_name_map[alias]
        layer_names.append(layer_name)
        name_alias_map[layer_name] = alias
    return layer_names, name_alias_map



def set_freeze_by_names(model, layer_alias=None, layer_names=None, freeze=True):
    # if not isinstance(layer_names, Iterable):
    #     layer_names = [layer_names]

    if layer_names is None and layer_alias is not None:
        layer_names, name_alias_map = get_actual_layer_names(model, layer_alias)
    elif layer_names is None and layer_alias is None:
        logging.info(f"No layer_names ")
        raise NotImplementedError
    else:
        pass

    logging.info(f"layer_names: {layer_names}")
    # for name, child in model.named_children():
    for name, module in model.named_modules():
        if name not in layer_names:
            continue
        else:
            logging.info(f"module: {module} is {'freezed' if freeze else 'NOT freezed'}")
            for param in module.parameters():
                param.requires_grad = not freeze


def freeze_by_names(model, layer_alias=None, layer_names=None):
    set_freeze_by_names(model, layer_alias=layer_alias, layer_names=layer_names, freeze=True)

def unfreeze_by_names(model, layer_alias=None, layer_names=None):
    set_freeze_by_names(model, layer_alias=layer_alias, layer_names=layer_names, freeze=False)


def get_modules_by_names(model, layer_alias=None, layer_names=None):
    if layer_names is None and layer_alias is not None:
        layer_names, name_alias_map = get_actual_layer_names(model, layer_alias)
    elif layer_names is None and layer_alias is None:
        logging.info(f"No layer_names ")
        raise NotImplementedError
    else:
        pass

    module_dict = {}
    logging.info(f"layer_names: {layer_names}")
    for name, module in model.named_modules():
        if name not in layer_names:
            continue
        else:
            module_dict[name_alias_map[name]] = module
            logging.info(f"Add module: {module} into module_dict")
    return module_dict


# def set_freeze_by_idxs(model, idxs, freeze=True):
#     if not isinstance(idxs, Iterable):
#         idxs = [idxs]
#     # num_child = len(list(model.children()))
#     # idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
#     # for idx, child in enumerate(model.children()):
#     #     if idx not in idxs:
#     #         continue
#     #     for param in child.parameters():
#     #         param.requires_grad = not freeze
#     num_params = len(list(model.named_parameters()))
#     idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
#     for idx, child in enumerate(model.children()):
#         if idx not in idxs:
#             continue
#         for param in child.parameters():
#             param.requires_grad = not freeze

# def freeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, True)

# def unfreeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, False)












