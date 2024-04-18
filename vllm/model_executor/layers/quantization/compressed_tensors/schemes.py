from abc import ABC, abstractmethod

class CompressedTensorsScheme(ABC):
    @abstractmethod
    def create_weights(self):
        raise NotImplementedError
    
    @abstractmethod
    def apply_weights(self):
        raise NotImplementedError



class CompressedTensorsW8A8StaticTensor(CompressedTensorsScheme):
    def create_weights(self):
        pass 
    
    def apply_weights(self):
        pass
        
