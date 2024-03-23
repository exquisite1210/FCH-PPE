from abc import ABC, abstractmethod
import copy

class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model1, model2,model3,model4,args=None):
        self.model1 = model1
        self.model2 = model2
        self.id = 0
        self.args = args
        self.model3 = model3
        self.dimZ = int(self.args.bit/2)
        self.alpha = 0
        self.src_model = copy.deepcopy(model3)
        self.src_model.freeze_grad()
        self.model4 = model4



    def set_id(self, trainer_id):
        self.id = trainer_id
    
    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters1,model_parameters2,model_parameters3):
        pass

    # @abstractmethod
    # def train(self, train_data, device, args=None):
    #     pass

    # @abstractmethod
    # def test(self, test_data, device, args=None):
    #     pass

    # @abstractmethod
    # def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
    #     pass


