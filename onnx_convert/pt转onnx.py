import torch
device = torch.device("cpu")

model = torch.load(r"C:\Users\xuzeran\PycharmProjects\PyTorch_YOLOv4\runs\exp55\weights\best.pt") # pytorch模型加载
batch_size = 1  #批处理大小
input_shape = (3, 1920, 1920)   #输入数据,改成自己的输入shape
model=model['model']
model=model.to(device)
# #set the model to inference mode
model.eval()
model.fuse()
x = torch.randn(batch_size, *input_shape,device='cpu')   # 生成张量
# x = x.to(device).cuda()
y = model(x,augment=False)  # dry run

export_onnx_file = r"C:\Users\xuzeran\PycharmProjects\PyTorch_YOLOv4\runs\exp55\weights\best1.onnx"	# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=12,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output",'train_output1','train_output2','train_output3'],	# 输出名
                  )
                  #   dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                  #                   "output":{0:"batch_size"}})

# import torch
# import torchvision
#
# dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
# model = torchvision.models.alexnet(pretrained=True).cuda()
#
# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]
#
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
