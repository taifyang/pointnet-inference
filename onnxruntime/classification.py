import numpy as np
import onnxruntime


point_num = 1024

 
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == '__main__':
    file = './bed_0610.txt'
    data = np.loadtxt(file, delimiter=',').astype(np.float32)
    point_set = data[:, 0:3]
    point_set = point_set[0:point_num, :]     
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)

    onnx_session = onnxruntime.InferenceSession("best_model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name=[]
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name=[]
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    input_feed={}
    for name in input_name:
        input_feed[name] = points

    pred = onnx_session.run(None, input_feed)[0]
    print(np.argmax(pred))
