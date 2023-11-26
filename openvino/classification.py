import numpy as np
from openvino.inference_engine import IECore


point_num = 1024


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == '__main__':
    data = np.loadtxt('./bed_0610.txt', delimiter=',')
    point_set = data[:, 0:3]
    point_set = point_set[0:point_num, :]     
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)

    ie = IECore()
    #net = ie.read_network(model="cls.onnx")
    net = ie.read_network(model="cls/cls_fp16.xml", weights="cls/cls_fp16.bin")
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_name = next(iter(net.input_info))
    infer_request_handle = exec_net.start_async(request_id=0, inputs={input_name: points})
    
    if infer_request_handle.wait(-1) == 0:
        output_layer = infer_request_handle._outputs_list[1]
        outputs = infer_request_handle.output_blobs[output_layer] 
        print(outputs.buffer)
        print(np.argmax(outputs.buffer))