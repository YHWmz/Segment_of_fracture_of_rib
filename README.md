# 医学3D图像检测分割
SJTU AI2612 Final HomeWork

小组成员：王宇昊 陈星元

Unet.py：模型文件

metric.py：包含了loss函数以及用于计算模型性能指标的函数

model_weights.pth：已经训练好的模型参数

preprocess.py：包含了用于数据预处理的函数和类

training.py：用于模型训练，运行完后会生成model_weights.pth文件，只需修改第24行与第26行的训练数据路径即可运行。

predict.py：用训练好的模型进行预测，输出csv以及nill文件。只需要修改第122、124与126行的路径即可运行。

