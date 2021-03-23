import numpy as np
from openvino.inference_engine import IECore

# node_feats = torch.rand(5, 92)
# edge_feats = torch.rand(60, 41)
node_feats = np.load("/home/anna/projects/deepchem/node_feats.npy")
edge_feats = np.load("/home/anna/projects/deepchem/edge_feats.npy")

### Run model OV ###
ie = IECore()
net = ie.read_network('/home/anna/projects/deepchem/model_cgcnn.xml')
exec_net = ie.load_network(net, 'CPU')

predictions = exec_net.infer(inputs={'input.1':node_feats, 'val.2':edge_feats})

print('OUTPUT_BLOBS')
print(predictions) # {'400': array([[nan]], dtype=float32)}

### Run model ONNX RUNTIME ###
import onnxruntime

session = onnxruntime.InferenceSession('/home/anna/projects/deepchem/model_cgcnn.onnx')
ort_inputs = {session.get_inputs()[0].name: node_feats, session.get_inputs()[1].name: edge_feats}
ort_outs = session.run(None, ort_inputs)

print('ONNX RUNTIME RES')
print(ort_outs)
