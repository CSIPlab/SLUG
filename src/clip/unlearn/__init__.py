from .calc_grad import calc_grad
from .FT import FT
from .GA import GA
from .GAFT import GAFT
from .SSD import SSD


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif "calc_grad" in name or "calc_importance" in name:
        return calc_grad
    elif name == "ft":
        return FT
    elif name == "ga" or name == "ga_o":
        return GA
    elif name == "gaft" or name == "gaft_o":
        return GAFT
    elif name == "salun" or name == "salun_o":
        return GAFT
    elif name == "ssd" or name == "ssd_o":
        return SSD
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
