import functools








def asset(_func):
    def decorator_asset(func):
        @functools.wraps(func)
        def wrapper_asset(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper_asset

    if _func is None:
        return decorator_asset
    else:
        return decorator_asset(_func)
