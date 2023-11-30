from base_layer import LayerBase


class CBowLayer(LayerBase):

    def __init__(self):
        super().__init__()
        self.layers = []


    def forward(self, context, target):
        pass

    def backward(self, dout):
        pass
