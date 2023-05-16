import torch
import pointnet_part_seg


point_num = 2048
class_num = 16
part_num = 50
normal_channel = False

def to_categorical(y, class_num):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(class_num)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

model = pointnet_part_seg.get_model(part_num, normal_channel)
#model = model.cuda() #cpu版本需注释此句
model.eval()
checkpoint = torch.load('./part_seg.pth')
model.load_state_dict(checkpoint['model_state_dict'])

x = (torch.rand(1, 6, point_num) if normal_channel else torch.rand(1, 3, point_num))
#=x = x.cuda() #cpu版本需注释此句
label = torch.randint(0, 1, (1, 1))
#label = label.cuda() #cpu版本需注释此句

export_onnx_file = "./part_seg.onnx"			
torch.onnx.export(model,
                    (x, to_categorical(label, class_num)),
                    export_onnx_file,              
                    opset_version = 10,
                    do_constant_folding = True,	
                    input_names = ["input"],		
                    output_names = ["output"],	
                    dynamic_axes = {"input":{0:"batch_size"},
                                    "output":{0:"batch_size"}})