import numpy as np
import onnxruntime


point_num = 2048
class_num = 1

 
def to_categorical(y, class_num):
    """ 1-hot encodes a tensor """
    new_y = np.eye(class_num)[y,]
    return new_y.astype(np.float32)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


if __name__ == '__main__':
    file = '85a15c26a6e9921ae008cc4902bfe3cd.txt'
    data = np.loadtxt(file).astype(np.float32)
    point_set = data[:, 0:3]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    choice = np.random.choice(point_set.shape[0], point_num, replace=True)
    point_set = point_set[choice, :][:, 0:3]
    pts = point_set

    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)
    label = np.array([[0]], dtype=np.int32)

    onnx_session = onnxruntime.InferenceSession("best_model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name=[]
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name=[]
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    input_feed={}
    input_feed[input_name[0]] = points
    input_feed[input_name[1]] = to_categorical(label, class_num)

    pred = onnx_session.run(None, input_feed)[0]

    cur_pred_val_logits = pred
    cur_pred_val = np.zeros((1, point_num)).astype(np.int32)
    
    logits = cur_pred_val_logits[0, :, :]
    cur_pred_val[0, :] = np.argmax(logits, 1)

    pts = np.append(points.reshape(point_num, 3), cur_pred_val[0, :].reshape(point_num, 1), 1)
    np.savetxt('pred.txt', pts, fmt='%.06f')       
