import inspect



# signature = inspect.signature(image)
# # print(signature.parameters)
# model_param = signature.parameters['model']
# base_cls = model_param._annotation.func_or_class().__class__.mro()[1]
# # str(base_cls) == str(ImageModel.__class__)
# str(base_cls)
# print(ImageModel.func_or_class().__class__)
# print(model_param._annotation.func_or_class().__class__)
# ImageModel.func_or_class().__class__ == model_param._annotation.func_or_class().__class__
# # print('sadf')


class Graph:


    def __init__(self, assets=None, pipes=None, models=None, prompts=None):
        
        self.assets = {a.__name__:a  for a in assets} if assets else {}
        self.pipes = {a.__name__:a  for a in pipes} if pipes else {}
        self.models = {a.__name__:a  for a in models} if models else {}
        self.prompts = {a.__name__:a  for a in prompts} if prompts else {}
        self.nodes = {}
        self.nodes.update(self.assets)
        self.nodes.update(self.pipes)
        self.nodes.update(self.models)
        self.nodes.update(self.prompts)
        
        for k, asset_func in self.assets.items():
            signature = inspect.signature(asset_func)
            func_kwargs = signature.parameters
            print(func_kwargs)



    def start(self):
        
        pass