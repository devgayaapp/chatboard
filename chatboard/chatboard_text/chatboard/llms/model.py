import functools
from components.image.image import Image
from ray import serve


# def model(_cls=None):
#     def decorator_model(cls):
#         @functools.wraps(cls)
#         def wrapper_model(*args, **kwargs):
#             return cls(*args, **kwargs)
#         return wrapper_model

#     if _cls is None:
#         return decorator_model
#     else:
#         return decorator_model(_cls)



def model(_cls=None):
    def decorator_model(cls):
        @serve.deployment()
        class ModelDeployment(cls):
            pass
        return ModelDeployment
    
    return decorator_model(_cls)



