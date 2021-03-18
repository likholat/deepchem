# import pytest

import deepchem as dc

from deepchem.molnet import load_perovskite
from openvino.inference_engine import IECore, ExecutableNetwork
from os import path
import torch

def test_model():
    node_feats = torch.rand(5, 92)
    edge_feats = torch.rand(60, 41)
    inputs = [node_feats, edge_feats]

    ### Run model ###
    ie = IECore()
    net = ie.read_network(model='/home/anna/projects/deepchem/model_cgcnn.xml', weights='/home/anna/projects/deepchem/model_cgcnn.bin')
    exec_net = ie.load_network(net, 'CPU')

    out_name = next(iter(exec_net.outputs.keys()))
    inp_name = []
    for name in list(exec_net.input_info.keys()):
        inp_name.append(name)
    
    predictions = exec_net.infer(inputs=dict(zip(inp_name, inputs)))

    print('OUTPUT_BLOBS')
    print(predictions) # {'400': array([[nan]], dtype=float32)}
