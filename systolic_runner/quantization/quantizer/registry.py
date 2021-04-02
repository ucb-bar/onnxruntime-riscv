from .quant_utils import QuantizationMode
from .operators.base_operator import QuantOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul
from .operators.attention import AttentionQuant
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
#from .operators.scan import Scan
from .operators.gather import GatherQuant
from .operators.conv import QLinearConv, ConvInteger, QLinearConvTranspose
from .operators.activation import QLinearActivation
from .operators.binary_op import QLinearBinaryOp
from .operators.maxpool import QMaxPool
from .operators.averagepool import QAveragePool
from .operators.reshape import QNoop
from .operators.reshape import QShape
from .operators.reshape import QScatter

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
#    "Scan": Scan
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "Conv": QLinearConv,
    "ConvTranspose": QLinearConvTranspose,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu" : QLinearActivation,
    "Sigmoid" : QLinearActivation,
    # I don't remember why I commented this out
    # IIRC it didn't affect accuracy _too_ much, but you can play around with it
    # "AveragePool": QAveragePool,
    "MaxPool": QMaxPool,
    "Reshape": QNoop,
    "Shape": QShape,
    "Size": QShape,
    "Transpose": QNoop,
    "Flatten": QNoop,
    "Resize": QNoop,
    "ScatterElements": QScatter,
    "Unsqueeze": QNoop,
    "Squeeze": QNoop,
    "Tile": QNoop,
    "Attention": AttentionQuant,
}
QLinearOpsRegistry.update(CommonOpsRegistry)


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)