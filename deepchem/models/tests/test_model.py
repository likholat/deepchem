
import torch
from openvino.inference_engine import IECore, ExecutableNetwork

ie = IECore()
net = ie.read_network('/home/anna/projects/deepchem/model_cgcnn.onnx')

# layer = 'Multiply_83'
import ngraph as ng
function = ng.function_from_cnn(net)

# nodes = []
# for node in function.get_ordered_ops():
#     try:
#         int(node.get_friendly_name())
#     except:
#         if not 'Const' in node.get_friendly_name():
#             nodes.append(node.get_friendly_name())
#             print(node.get_friendly_name())

# net.add_outputs(nodes)

exec_net = ie.load_network(net, 'CPU')

out_name = []
for name in list(exec_net.outputs.keys()):
    out_name.append(name)

inp_name = []
for name in list(exec_net.input_info.keys()):
    inp_name.append(name)

node_feats = torch.randn(92)
edge_feats = torch.randn(41)

predictions = exec_net.infer(inputs={inp_name[0]:node_feats, inp_name[1]:edge_feats})

output1 = predictions[out_name[0]]
output = predictions[out_name[1]]

print(output1)
print(output)

# import numpy as np
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# for node in function.get_ordered_ops():
#     try:
#         res = predictions[node.get_friendly_name()]
#         if (np.isinf(res).any() | np.isnan(res).any()):
#             print('WRONG VALUE') # <MatMul: 'MatMul_498' ({1,128})>
#             print(node)
#             print(res)
#             break
#     except KeyError as e:
#         print('Wrong key - ' + str(e))

# import numpy as np
# np.savetxt('out.txt', predictions[layer])

# print('OUTPUT_BLOBS')
# print(predictions) # {'400': array([[nan]], dtype=float32)}


# ### Run model ###
# ie = IECore()
# net = ie.read_network('/home/anna/projects/deepchem/torch_model.onnx')
# exec_net = ie.load_network(net, 'CPU')

# out_name = next(iter(exec_net.outputs.keys()))
# inp_name = next(iter(exec_net.input_info.keys()))

# inp = torch.randn([8, 1024])

# predictions = exec_net.infer(inputs={inp_name: inp})

# print('OUTPUT_BLOBS')
# print(predictions) # {'7': array([[ 0.15030602],
#                             # [-0.00887963],
#                             # [ 0.02521402],
#                             # [-0.01307261],
#                             # [-0.30070552],
#                             # [-0.27964684],
#                             # [-0.49185547],
#                             # [-0.04043305]], dtype=float32)}
