import torch
from openvino.inference_engine import IECore, ExecutableNetwork

torch.manual_seed(124)

node_feats = torch.rand(5, 92)
edge_feats = torch.rand(60, 41)
inputs = [node_feats, edge_feats]

### Run model OV ###
ie = IECore()
net = ie.read_network('/home/anna/projects/deepchem/model_cgcnn.onnx')
exec_net = ie.load_network(net, 'CPU')

out_name = next(iter(exec_net.outputs.keys()))
inp_name = []
for name in list(exec_net.input_info.keys()):
    inp_name.append(name)

predictions = exec_net.infer(inputs=dict(zip(inp_name, inputs)))

print('OUTPUT_BLOBS')
print(predictions) # {'400': array([[nan]], dtype=float32)}

### Run model ONNX RUNTIME ###
import onnxruntime

session = onnxruntime.InferenceSession('/home/anna/projects/deepchem/model_cgcnn.onnx')
ort_inputs = {session.get_inputs()[0].name: node_feats.detach().numpy(), session.get_inputs()[1].name: edge_feats.detach().numpy()}
ort_outs = session.run(None, ort_inputs)

print('ONNX RUNTIME RES')
print(ort_outs) # [array([[25.652527]], dtype=float32)]
