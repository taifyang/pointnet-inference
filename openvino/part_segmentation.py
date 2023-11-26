import numpy as np
from openvino.inference_engine import IECore


point_num = 2048
class_num = 16

 
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
    data = np.loadtxt('85a15c26a6e9921ae008cc4902bfe3cd.txt')
    point_set = data[:, 0:3]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    choice = np.random.choice(point_set.shape[0], point_num, replace=True)
    point_set = point_set[choice, :][:, 0:3]
    pts = point_set

    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)
    label = np.array([[0]], dtype=np.int32)

    ie = IECore()
    #net = ie.read_network(model="part_seg.onnx")
    net = ie.read_network(model="part_seg/part_seg_fp16.xml", weights="part_seg/part_seg_fp16.bin")
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_names = []
    for key in net.input_info:
        input_names.append(key)
    infer_request_handle=exec_net.start_async(request_id=0, inputs={input_names[0]:to_categorical(label, class_num), input_names[1]:points})
    
    if infer_request_handle.wait(-1) == 0:
        output_layer = infer_request_handle._outputs_list[1]
        outputs = infer_request_handle.output_blobs[output_layer] 

        cur_pred_val_logits = outputs.buffer
        cur_pred_val = np.zeros((1, point_num)).astype(np.int32)  
        logits = cur_pred_val_logits[0, :, :]
        cur_pred_val[0, :] = np.argmax(logits, 1)

        pts = np.append(pts.reshape(point_num, 3), cur_pred_val[0, :].reshape(point_num, 1), 1)
        np.savetxt('pred.txt', pts, fmt='%.06f')       