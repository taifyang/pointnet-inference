import numpy as np
import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda  


point_num = 2048
class_num = 16
parts_num = 50

 
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
    logger = trt.Logger(trt.Logger.WARNING)
    with open("part_seg.trt", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    h_input0 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_input1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    h_output0 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
    h_output1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(3)), dtype=np.float32)
    d_input0 = cuda.mem_alloc(h_input0.nbytes)
    d_input1 = cuda.mem_alloc(h_input1.nbytes)
    d_output0 = cuda.mem_alloc(h_output0.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    stream = cuda.Stream()
    
    data = np.loadtxt('85a15c26a6e9921ae008cc4902bfe3cd.txt').astype(np.float32)
    point_set = data[:, 0:3]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    choice = np.random.choice(point_set.shape[0], point_num, replace=True)
    point_set = point_set[choice, :][:, 0:3]
    pts = point_set

    points = np.reshape(point_set, ((1, point_num, 3)))
    points = points.swapaxes(2, 1)
    label = np.array([[0]], dtype=np.int32)

    np.copyto(h_input0, points.ravel())
    np.copyto(h_input1,  to_categorical(label, class_num).ravel())

    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input0, h_input0, stream)
        cuda.memcpy_htod_async(d_input1, h_input1, stream)
        context.execute_async_v2(bindings=[int(d_input0), int(d_input1),int(d_output0), int(d_output1)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
        cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
        stream.synchronize()

        cur_pred_val_logits = h_output1.reshape(1, point_num, parts_num)
        cur_pred_val = np.zeros((1, point_num)).astype(np.int32)
        
        logits = cur_pred_val_logits[0, :, :]
        cur_pred_val[0, :] = np.argmax(logits, 1)

        pts = np.append(pts.reshape(point_num, 3), cur_pred_val[0, :].reshape(point_num, 1), 1)
        np.savetxt('pred.txt', pts, fmt='%.06f')       
