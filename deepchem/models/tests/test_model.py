import deepchem as dc

from deepchem.molnet import load_perovskite
from openvino.inference_engine import IECore, ExecutableNetwork
from os import path

def test_model():
    ### Create inputs ###

    current_dir = path.dirname(path.abspath(__file__))
    config = {
        "reload": False,
        "featurizer": dc.feat.CGCNNFeaturizer(),
        # disable transformer
        "transformers": [],
        "data_dir": current_dir
    }
    
    tasks, datasets, _ = load_perovskite(**config)
    train, _, _ = datasets

    n_tasks = len(tasks)
    torch_model = dc.models.CGCNNModel(
        n_tasks=n_tasks,
        mode='regression',
        batch_size=1,
        learning_rate=0.001)

    generator = torch_model.default_generator(
        train, mode='regression', pad_batches=False)
  
    for inp_id, batch in enumerate(generator):
        inputs, labels, weights = batch
        inputs, _, _ = torch_model._prepare_batch((inputs, None, None))


    node_feats = inputs.ndata.pop('x')
    edge_feats = inputs.edata.pop('edge_attr')
    inputs = [node_feats, edge_feats]

    ### Run model ###
    ie = IECore()
    net = ie.read_network('model_cgcnn.onnx')
    exec_net = ie.load_network(net, 'CPU')

    out_name = next(iter(exec_net.outputs.keys()))
    inp_name = []
    for name in list(exec_net.input_info.keys()):
        inp_name.append(name)
    
    predictions = exec_net.infer(inputs=dict(zip(inp_name, inputs)))

    print('OUTPUT_BLOBS')
    print(predictions) # {'400': array([[nan]], dtype=float32)}
