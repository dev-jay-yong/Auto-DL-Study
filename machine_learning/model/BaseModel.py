import BaseModel


class OptimizerObject(BaseModel):
    function_name: str = "Adam"
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: None
    decay: float = 0.0
    amsgrad: False


class HyperParamObject(BaseModel):
    epochs: int = 50
    loss_function: str = "mean_squared_error"
    optimizer: dict = OptimizerObject
    activation: str
    batch_size: int
    is_deep_learning: bool
    layer_width: int
    layer_deep: int

    def __dict__(self):
        self = self.__dict__['__object__']
