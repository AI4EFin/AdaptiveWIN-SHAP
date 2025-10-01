import inspect
import functools

def store_init_kwargs(_fn=None, *, to_attr: str = "init_kwargs", normalize_device: bool = True):
    """
    Decorator for __init__ that captures the exact kwargs used to construct the object.
    Works on subclasses, supports positional/keyword args, and applies __init__ defaults.
    Usage:
        @store_init_kwargs
        def __init__(...): ...
    or
        @store_init_kwargs(to_attr="constructor", normalize_device=False)
        def __init__(...): ...
    """
    def _decorator(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(init)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            # Drop `self`
            captured = {k: v for k, v in bound.arguments.items() if k != "self"}

            if normalize_device:
                try:
                    import torch
                    for k, v in list(captured.items()):
                        if isinstance(v, torch.device):
                            captured[k] = str(v)  # e.g., "cpu", "cuda:0", "mps"
                except Exception:
                    pass

            # Store a shallow copy to avoid mutation surprises
            setattr(self, to_attr, dict(captured))
            return init(self, *args, **kwargs)
        return wrapper

    # Allow both @store_init_kwargs and @store_init_kwargs(...)
    if _fn is not None and callable(_fn):
        return _decorator(_fn)
    return _decorator


def new_with_same_init(obj, **overrides):
    """
    Convenience: re-instantiates `obj`'s class with the captured init kwargs,
    optionally overriding some of them.
    """
    if not hasattr(obj, "init_kwargs"):
        raise AttributeError("Object has no `init_kwargs`. Did you decorate __init__ with @store_init_kwargs?")
    kw = dict(obj.init_kwargs)
    kw.update(overrides)
    return type(obj)(**kw)
