from collections import defaultdict
from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
ONNX_ML = False
def should_render_domain(domain):  # type: (Text) -> bool
    if domain == ONNX_ML_DOMAIN and not ONNX_ML:
        return False
    elif ONNX_ML and domain != ONNX_ML_DOMAIN:
        return False
    return True

def get_native_int8_ops():
    native_int8_ops = []
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
    exsting_ops = set()  # type: Set[Text]
    for domain, _supportmap in sorted(index.items()):
        if not should_render_domain(domain):
            continue

        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in exsting_ops:
                    continue
                exsting_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))

    for domain, supportmap in operator_schemas:
        for _, namemap in supportmap:
            for op_type, schema, versions in namemap:
                if schema.type_constraints and 'tensor(int8)' in schema.type_constraints[0].allowed_type_strs: 
                    native_int8_ops.append(op_type)
    return native_int8_ops
