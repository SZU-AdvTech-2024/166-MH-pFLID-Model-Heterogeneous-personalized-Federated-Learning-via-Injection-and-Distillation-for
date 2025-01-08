import torch  # 导入 PyTorch 库
import torchvision
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块，用于构建模型
import torch.optim as optim  # 导入 PyTorch 中的优化器模块，用于优化模型
from torch.utils.data import Dataset, TensorDataset
import numpy as np  # 导入 NumPy 用于数据处理
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from PIL import Image  # 用于将图像分辨率进行下采样
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import random_split, DataLoader, TensorDataset
import csv
import matplotlib.pyplot as plt
import random
import shutil

from DataReader import readData
import argparse

"""
    Server端，Messenger，localModel，Receiver，Transmitter
"""


# Server端，控制信使模型的聚合以及，更新。
class Server(nn.Module):
    def __init__(self, num_classes=2, num_clients=4):
        super(Server, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_clients = num_clients  # 客户端的个数
        # 用于存储聚合后的 body 和 head 的参数。
        self.aggregated_body = None
        self.aggregated_head = None

        # 用于存储上传的Messenger模型的 body 和 head 部分的参数
        self.uploaded_messenger_body = []  # 初始化为一个空列表
        self.uploaded_messenger_head = []  # 初始化为一个空列表

    # 用于上传客户端的messenger_model的参数到Server端。
    def receive_client_models_params(self, messengers):
        """
        接收所有客户端的 messenger_model 的 body 和 head 参数，
        并分别存储在 uploaded_messenger_body 和 uploaded_messenger_head 中。
        :param messengers: 一个包含多个客户端 Messenger 模型的列表
        """
        # 清空之前的 uploaded_messenger_body 和 uploaded_messenger_head
        self.uploaded_messenger_body.clear()
        self.uploaded_messenger_head.clear()

        # 遍历每个客户端的信使模型，将其 body 和 head 部分的参数添加到上传列表中
        for messenger in messengers:
            # 获取该模型的 body 和 head 部分参数
            body_params = list(messenger.body.parameters())
            head_params = list(messenger.head.parameters())

            # 存储 body 和 head 部分的参数
            self.uploaded_messenger_body.append(body_params)
            self.uploaded_messenger_head.append(head_params)

    # 聚合Messenger模型的body参数部分
    def aggregate_body(self, messengers, weights=None):
        """
        聚合多个 Messenger 模型的 body 参数部分。
        :param messengers: 一个包含多个 Messenger 模型的列表
        :param weights: 每个 Messenger 模型的权重 (例如根据训练样本数来设定)
        :return: 聚合后的 body 参数
        """
        # 如果没有提供权重，则默认所有模型权重相同
        if weights is None:
            weights = [1.0] * len(messengers)  # 默认每个模型的权重为 1

        total_weight = sum(weights)

        # 聚合 body 部分的参数
        self.aggregated_body = self.aggregate_parameters([m.body for m in messengers], weights, total_weight)

        return self.aggregated_body

    # 聚合Messenger模型的head参数部分,如果权重没有给出，则按照每个权重相同进行。
    def aggregate_head(self, messengers, weights=None):
        """
        聚合多个 Messenger 模型的 head 参数部分。
        :param messengers: 一个包含多个 Messenger 模型的列表
        :param weights: 每个 Messenger 模型的权重 (例如根据训练样本数来设定)
        :return: 聚合后的 head 参数
        """
        # 如果没有提供权重，则默认所有模型权重相同
        if weights is None:
            weights = [1.0] * len(messengers)  # 默认每个模型的权重为 1

        total_weight = sum(weights)

        # 聚合 head 部分的参数
        self.aggregated_head = self.aggregate_parameters([m.head for m in messengers], weights, total_weight)

        return self.aggregated_head

    # 聚合操作,输入的是待聚合模型的部分层。
    def aggregate_parameters(self, model_parts, weights, total_weight):
        """
        对模型的不同部分 (body 或 head) 进行参数聚合。
        :param model_parts: 一个包含多个模型部分的列表（比如多个 body 或多个 head）
        :param weights: 权重列表
        :param total_weight: 权重总和
        :return: 聚合后的参数
        """
        aggregated_params = []

        # 遍历每一层，进行加权平均
        for i in range(len(list(model_parts[0].parameters()))):  # 确保以第一个模型的参数数量为准
            layer_params = []
            # 遍历所有模型
            for j in range(len(model_parts)):
                params = list(model_parts[j].parameters())
                # 检查每个模型的参数数量和当前参数的索引
                # print(f"Model {j} body has {len(params)} parameters.")
                if i >= len(params):  # 如果当前模型的参数数量少于第一个模型
                    raise ValueError(f"Model {j} does not have parameter at index {i}")

                param = params[i]  # 获取该模型的第 i 层参数
                layer_params.append(param * weights[j])  # 对参数进行加权

        # 生成一个新的聚合后的模型部分
        aggregated_model = nn.ModuleList([layer for layer in aggregated_params])
        return aggregated_model

    # 更新所有Messenger模型的body参数。
    def update_body(self, messengers):
        """
        更新所有 Messenger 模型的 body 部分。
        :param messengers: 一个包含多个 Messenger 模型的列表
        """
        # 将聚合后的 body 参数传递给每个 Messenger 模型
        for messenger in messengers:
            self.update_parameters(messenger.body, self.aggregated_body)

    # 更新所有Messenger模型的head参数
    def update_head(self, messengers):
        """
        更新所有 Messenger 模型的 head 部分。
        :param messengers: 一个包含多个 Messenger 模型的列表
        """
        # 将聚合后的 head 参数传递给每个 Messenger 模型
        for messenger in messengers:
            self.update_parameters(messenger.head, self.aggregated_head)

    # 更新神经网络层参数,用于更新Messenger模型。
    def update_parameters(self, model_part, aggregated_params):
        """
        将聚合后的参数更新到模型的指定部分 (body 或 head)。
        :param model_part: 要更新的模型部分（例如 body 或 head）
        :param aggregated_params: 聚合后的参数
        """
        # 获取模型部分的参数
        model_params = list(model_part.parameters())

        # 遍历每一层，将聚合后的参数赋值给相应的模型层
        for param, aggregated_param in zip(model_params, aggregated_params):
            param.data.copy_(aggregated_param.data)


# Messenger模型
class Messenger(nn.Module):
    def __init__(self, input_size_mes, input_size_loc, output_size, num_classes=2):
        super(Messenger, self).__init__()
        self.device = device
        self.linear_input_size = input_size_mes
        # 定义 body 部分 (卷积层和池化层)
        self.body = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 输入通道3，输出通道64，卷积核大小3
            nn.ReLU(),  # 激活函数 ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化，2x2的池化窗口

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),  # 输入通道64，输出通道64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化，2x2的池化窗口

            nn.Conv2d(64, 512, kernel_size=7, stride=2, padding=1),  # 输入通道64，输出通道512
            nn.ReLU()
        )

        # 定义 head 部分 (全连接层),使用全连接层进行分类。
        self.head = nn.Sequential(
            nn.Flatten(),  # 展平层
            nn.Linear(self.linear_input_size, 256),  # 输入大小是 body 输出的特征数
            nn.BatchNorm1d(256),
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(256, num_classes)  # 最终输出分类结果
        )

        # 接收器模块
        self.receiver = Receiver(input_size_mes, input_size_loc, output_size)

        # 传输器模块
        self.transmitter = Transmitter(input_size_loc, input_size_mes, output_size)

        # stage用于控制选择哪种行为（知识注入或知识蒸馏）
        self.stage = "injection"  # 默认阶段为知识注入

    # 定义前向传播
    def forward(self, x, local_body_feature, current_stage="injection"):
        x = self.body(x)
        self.stage = current_stage
        if isAddFeature == True:
            # 在特征加阶段，逐元素加特征
            # 调整 x 的空间维度与 local_body_feature 一致
            local_body_feature = match_dimensions(x, local_body_feature)
            conv_layer = nn.Conv2d(local_body_feature.shape[1], 512, kernel_size=1).to(local_body_feature.device)
            local_body_feature = conv_layer(local_body_feature)  # 将 local_body_feature 的通道数调整为 512
            x = x + local_body_feature  # 特征逐元素相加

        else:
            # 根据当前阶段进行选择
            if self.stage == "injection":
                # 在知识注入阶段，使用Receiver模块
                x = self.receiver(x, local_body_feature)  # 使用Receiver产生预测

            elif self.stage == "distillation":
                # 在知识蒸馏阶段，使用Transmitter模块
                x = self.transmitter(local_body_feature, x)  # 使用Transmitter产生预测

        # 通过 head 部分 (全连接层)
        x = self.head(x)
        return x

    # 计算的是知识蒸馏阶段的总的loss
    def knowledge_distillation_loss(self, local_preds, messenger_preds, data_y, lambda_m=0.9, lambda_con=0.1):
        """
        知识蒸馏损失函数，计算Messenger模型和Local模型之间的损失。

        :param messenger_preds: 信使模型的预测值
        :param data_y: 真实标签（ground truth）
        :param local_preds: 本地模型的预测值 (Local model predictions)
        :param lambda_m: Messenger模型损失的权重系数
        :param lambda_con: 本地模型与Messenger模型之间对比损失的权重系数
        :return: 知识蒸馏的总损失值
        """
        # 1. 计算 Messenger 模型的预测值

        # 2. 计算Messenger模型的损失 L^m_dis
        criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数，用于分类任务,损失函数也移到GPU上
        messenger_loss = criterion(messenger_preds, data_y)  # Messenger模型的损失

        # 3. 计算Messenger模型与本地模型预测之间的对比损失 L^con_dis 按照原文最后的补充部分的定义计算。
        # 目标分布是local_preds，本地模型的输出应该经过softmax处理
        # 使用softmax将输出转换为概率分布
        epsilon = 1e-12  # 防止 log(0)
        local_preds_softmax = F.softmax(local_preds, dim=1)  # 本地模型的预测
        messenger_preds_softmax = F.softmax(messenger_preds, dim=1)  # Messenger模型的预测

        # 计算KL散度
        kl_loss = torch.sum(
            local_preds_softmax * torch.log(local_preds_softmax + epsilon) - messenger_preds_softmax * torch.log(
                messenger_preds_softmax + epsilon), dim=1).mean()

        # 4. 加权求和损失
        distillation_loss = lambda_m * messenger_loss + lambda_con * kl_loss  # 总损失

        return distillation_loss, messenger_loss, kl_loss

    # 加载模型
    def load_model(self, model_save_path):
        """
        加载指定路径的模型参数。
        :param model_save_path: 模型保存的路径
        """
        # 加载模型的 state_dict
        self.load_state_dict(torch.load(model_save_path))
        self.eval()  # 设置模型为评估模式
        print(f"Model loaded from {model_save_path}")


# local model，本地客户端模型，分为body和head两部分。
class LocalModel(nn.Module):
    def __init__(self, resnet_type='resnet17', num_classes=2, data_x=None, data_y=None, dataloader=None):
        super(LocalModel, self).__init__()
        self.device = device
        self.body_output_channels = 1  # body部分的输出通道数
        self.model_type = resnet_type  # 本地模型的类型
        self.model_save_path = resnet_type + "_only_local.pt"
        # 根据指定的 ResNet 类型，实例化不同的模型
        if resnet_type == 'resnet17':
            self.resnet = resnet17(num_classes)
            self.body_output_channels = self.resnet.get_body_output_channels_num()  # 获取body部分输出的通道数
        elif resnet_type == 'resnet11':
            self.resnet = resnet11(num_classes)
            self.body_output_channels = self.resnet.get_body_output_channels_num()
        elif resnet_type == 'resnet8':
            self.resnet = resnet8(num_classes)
            self.body_output_channels = self.resnet.get_body_output_channels_num()
        elif resnet_type == 'resnet5':
            self.resnet = resnet5(num_classes)
            self.body_output_channels = self.resnet.get_body_output_channels_num()
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # body 部分:
        self.body = self.resnet.body

        # head 部分:
        self.head = self.resnet.head

        # 数据存储:
        # 将加载的图片数据（data_x）和标签数据（data_y）作为模型属性
        self.data_x = data_x  # 图像数据
        self.data_y = data_y  # 图像对应的标签

        # 数据加载器，提供了图像数据和标签
        self.dataloader = dataloader

    # 定义前向传播
    def forward(self, x):
        # 通过 body 部分 (卷积和池化)
        x = self.forward_body(x)
        # 通过 head 部分 (全连接层)

        x = self.forward_head(x)
        return x

    # 定义body部分的前向传播，考虑有些层为None的情况
    def forward_body(self, x):
        for layer in self.body:
            if layer is not None:
                x = layer(x)  # 跳过None层
        return x

    def forward_head(self, x):
        # 通过 head 部分 (全连接层)
        for layer in self.head:
            if layer is not None:
                x = layer(x)  # 跳过None层
        return x

    # only local training,且保存模型
    def train_model(self, train_dataloader, num_epochs=20, learning_rate=0.0001):
        criterion = nn.CrossEntropyLoss().to(device)  # 损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # 优化器
        self.to(self.device)
        train_losses = []  # 保存每个epoch的损失
        train_accuracies = []  # 保存每个epoch的准确率

        for epoch in range(num_epochs):
            self.train()  # 设置模型为训练模式
            running_loss = 0.0
            correct = 0
            total = 0

            for data_x, data_y in train_dataloader:
                data_x, data_y = data_x.to(self.device), data_y.to(self.device)
                optimizer.zero_grad()  # 清空梯度
                outputs = self.forward(data_x)  # 前向传播
                loss = criterion(outputs, data_y)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新模型参数

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += data_y.size(0)
                correct += (predicted == data_y).sum().item()

            avg_loss = running_loss / len(train_dataloader)
            accuracy = correct / total

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 绘制损失和准确率曲线
        self.plot_metrics(train_losses, train_accuracies)
        # 保存训练后的模型
        self.save_model(self.model_save_path)

    # 保存模型
    def save_model(self, model_save_path):
        """
        保存模型参数到指定路径。
        :param model_save_path: 模型保存的路径
        """
        # 保存模型的 state_dict
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # 绘制only local model训练时的损失曲线以及acc曲线
    # 加载模型
    def load_model(self, model_save_path):
        """
        加载指定路径的模型参数。
        :param model_save_path: 模型保存的路径
        """
        # 加载模型的 state_dict
        self.load_state_dict(torch.load(model_save_path))
        self.eval()  # 设置模型为评估模式
        print(f"Messenger loaded from {model_save_path}")

    def plot_metrics(self, train_losses, train_accuracies):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(14, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy over Epochs")
        plt.legend()
        # 保存图像
        plt.savefig(f"{self.model_type}_training_metrics.png")

        # 显示图像 5 秒后关闭
        plt.show(block=False)  # 非阻塞模式显示图像
        plt.pause(5)  # 显示 5 秒
        plt.close()  # 关闭图像窗口

    # 定义知识注入阶段的损失函数,返回知识注入阶段的损失函数值
    def knowledge_injection_loss(self, messenger_preds, local_preds, targets, lambda_l=0.9, lambda_m=0.1):
        """
        计算知识注入的加权损失：L_inj = λ_l * L_local + λ_m * L_messenger
        :param messenger_preds: Messenger 模型的预测
        :param local_preds: local 模型的预测值
        :param targets: 真实标签
        :param lambda_l: 本地模型损失的权重
        :param lambda_m: Messenger 模型损失的权重
        :return: 计算得到的总损失
        """
        # 损失函数放置到GPU上
        local_criterion = nn.CrossEntropyLoss().to(device)  # 设置local损失函数为交叉熵函数,
        messenger_criterion = nn.CrossEntropyLoss().to(device)  # 设置Messenger模型的损失函数为交叉熵函数。
        # 计算本地模型的损失
        loss_local = local_criterion(local_preds, targets)  # 计算损失值

        # 计算 Messenger 模型的损失
        loss_messenger = messenger_criterion(messenger_preds, targets)
        # 给loss限制范围，不允许太大
        clamped_loss_messenger = torch.clamp(loss_messenger, max=max_loss2)
        # 返回加权总损失
        total_loss = lambda_l * loss_local + lambda_m * clamped_loss_messenger

        return total_loss, loss_local, loss_messenger

    # 加载数据
    def load_data(self, images, labels):
        """
        加载图片数据和标签数据，分别赋值给 data_x 和 data_y。

        :param images: 加载的图像数据
        :param labels: 加载的标签数据
        """
        self.data_x = images  # 输入图像数据
        self.data_y = labels  # 标签数据

    # 加载数据加载器
    def load_dataloader(self, dataloader=None):
        self.dataloader = dataloader

    # 获取LocalModel的数据
    def get_data(self):
        """
        获取当前加载的数据。
        :return: (data_x, data_y) 图像数据和标签
        """
        return self.data_x, self.data_y

    # 直接读取文件的数据,根据文件夹的路径，以及需要的放大倍数的图片
    def dataReader(self, folder_path, magnification="40X"):
        """
        读取指定文件夹路径下所有特定放大倍数（如 40X）的图片，并返回图片列表和图片数量。

        :param folder_path: 包含图片的文件夹路径
        :param magnification: 需要加载的图片放大倍数（默认为"40X"）
        :return: 加载的图片列表和图片数量
        """
        images = []  # 用于存储加载的图片
        images_y = []  # 用于存储图像的标签（0表示良性B，1表示恶性M）
        count_images = 0  # 用于统计加载的图片数量

        # 检查给定的文件夹路径是否存在
        if not os.path.exists(folder_path):
            print("提供的路径不存在！")
            return [], [], 0  # 路径不存在时，返回空列表和0数量

        # 遍历文件夹中的所有文件夹和子文件夹
        for root, dirs, files in os.walk(folder_path):  # 使用os.walk遍历目录树
            # 只选择包含指定放大倍数的文件夹
            if magnification in root:  # 如果路径中包含指定的放大倍数
                # 遍历该文件夹中的所有文件
                for filename in files:
                    # 构建文件的完整路径
                    file_path = os.path.join(root, filename)

                    # 检查文件是否为图片文件（例如 .png, .jpg, .jpeg, .bmp, .tiff 等格式）
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        try:
                            # 打开图片并转换为 RGB 模式
                            img = Image.open(file_path).convert('RGB')
                            self.data_x.append(img)  # 将加载的图片添加到列表中
                            count_images += 1  # 统计图片数量

                            # 通过文件名来提取肿瘤类型标签
                            file_name = filename.split('.')[0]  # 去掉扩展名，获取文件名部分
                            tumor_type = file_name.split('_')[1]  # 假设第二部分是肿瘤类型（B 或 M）

                            # 根据肿瘤类型给图像打标签：'B' = 良性，'M' = 恶性
                            if tumor_type == "B":
                                self.data_y.append(0)  # 0表示良性
                            elif tumor_type == "M":
                                self.data_y.append(1)  # 1表示恶性
                            else:
                                print(f"无法识别肿瘤类型: {tumor_type}，跳过该图片")

                        except Exception as e:
                            # 如果加载图片失败，输出错误信息
                            print(f"无法加载图片 {filename}: {e}")


# 接收器 receiver
class Receiver(nn.Module):
    def __init__(self, input_size_mes, input_size_loc, output_size):
        super(Receiver, self).__init__()
        # 在第一次前向传播的时候初始化
        # 定义查询、键、值的线性变换权重矩阵，这几行相当于按照Messenger的维度输出。
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.device = device  # 可以传入设备，若未传入则默认为 CPU
        # 用于标识是否是第一次执行前向传播
        self.first_forward = True
        self.cnt = 0

    def load_model(self, model_save_path):
        """
        加载指定路径的模型参数。
        :param model_save_path: 模型保存的路径
        """
        # 加载模型的 state_dict
        self.load_state_dict(torch.load(model_save_path))
        self.eval()  # 设置模型为评估模式
        print(f"Receiver loaded from {model_save_path}")

    def forward(self, I_mes, I_loc):
        batch_size, channels, H, W = I_mes.size()
        loc_batch_size, loc_channels, loc_H, loc_W = I_loc.size()
        # print(f"I_mes：{I_mes.shape}")
        # print(f"I_loc：{I_loc.shape}")

        # Step 1: 初始化各个线性层，仅在首次前向传播时
        if self.first_forward:
            self.W_d = nn.Linear(loc_channels * loc_H * loc_W, channels * H * W).to(device)  # I_loc 映射到 I_mes
            # 初始化 Q, K, V 的线性层
            self.W_q = nn.Linear(I_mes.size(1) * H * W, channels).to(device)
            self.W_k = nn.Linear(I_mes.size(1) * H * W, channels).to(device)
            self.W_v = nn.Linear(I_mes.size(1) * H * W, channels).to(device)

            self.first_forward = False  # 更新标志位，确保只在第一次前向传播时初始化

        # Step 2: 处理本地特征 I_loc，通过 W_d 映射到 I_loc'

        # 将 I_mes 和 I_loc 展平为 2D 张量
        I_loc_flat = I_loc.flatten(start_dim=1)
        I_mes_flat = I_mes.flatten(start_dim=1)  # 展平为 2d
        I_loc_prime = self.W_d(I_loc_flat)  # 处理本地特征 I_loc，通过 W_d 映射到 I_loc',2d张量

        # Step 4: 计算 Q, K, V
        Q = self.W_q(I_mes_flat)
        K = self.W_k(I_loc_prime)
        V = self.W_v(I_loc_prime)

        # Step 5: 计算注意力得分矩阵 M_R
        d_mes = Q.size(-1)  # 查询向量的维度
        M_R = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(d_mes, dtype=torch.float32))  # QK^T / sqrt(d_mes)
        M_R = F.softmax(M_R, dim=-1)  # Softmax 归一化
        # Step 6: 计算加权本地特征 I_loc_R
        I_loc_R = torch.matmul(M_R, V)  # I_loc,R = M_R * V
        return I_loc_R


# 传输器 transmitter
class Transmitter(nn.Module):
    def __init__(self, input_size_loc, input_size_mes, output_size):
        super(Transmitter, self).__init__()
        self.device = device
        # 定义线性层 W_d 用于将本地特征 I_loc 映射到与 I_mes 相同的维度
        self.W_d = None
        # 定义线性层生成 Q, K 和 V
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.first_forward = True

    def load_model(self, model_save_path):
        """
        加载指定路径的模型参数。
        :param model_save_path: 模型保存的路径
        """
        # 加载模型的 state_dict
        self.load_state_dict(torch.load(model_save_path))
        self.eval()  # 设置模型为评估模式
        print(f"Transmitter loaded from {model_save_path}")

    def forward(self, I_loc, I_mes):
        """
        I_loc_prime: 本地特征经过处理后的特征 (Iloc')
        I_mes: 全局特征 (Imes)
        """
        batch_size, channels, H, W = I_mes.size()
        loc_batch_size, loc_channels, loc_H, loc_W = I_loc.size()

        # Step 1: 初始化各个线性层，仅在首次前向传播时
        if self.first_forward:
            self.W_d = nn.Linear(loc_channels * loc_H * loc_W, channels * H * W).to(device)  # I_loc 映射到 I_mes
            # 初始化 Q, K, V 的线性层
            self.W_q = nn.Linear(I_mes.size(1) * H * W, channels).to(device)
            self.W_k = nn.Linear(I_mes.size(1) * H * W, channels).to(device)
            self.W_v = nn.Linear(I_mes.size(1) * H * W, channels).to(device)

            self.first_forward = False  # 更新标志位，确保只在第一次前向传播时初始化
        # Step 2: 处理本地特征 I_loc，通过 W_d 映射到 I_loc'

        # 将 I_mes 和 I_loc 展平为 2D 张量
        I_loc_flat = I_loc.flatten(start_dim=1)
        I_mes_flat = I_mes.flatten(start_dim=1)

        I_loc_prime = self.W_d(I_loc_flat)  # 处理本地特征 I_loc，通过 W_d 映射到 I_loc',2d张量

        # Step 3: 计算 Q, K, V，输入形状均为 `[batch_size, mes_channels * mes_height * mes_width]`
        Q = self.W_q(I_loc_prime)
        K = self.W_k(I_mes_flat)
        V = self.W_v(I_mes_flat)

        # Step 4: 计算注意力得分矩阵 M_R
        d_mes = Q.size(-1)  # 查询向量的维度
        M_R = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(d_mes, dtype=torch.float32))  # QK^T / sqrt(d_mes)
        M_R = F.softmax(M_R, dim=-1)  # Softmax 归一化
        # Step 6: 计算加权本地特征 I_loc_R
        I_mes_T = torch.matmul(M_R, V)  # I_loc,R = M_R * V

        return I_mes_T


"""
    知识注入、知识蒸馏、聚合、参数更新，以及通信
"""


# 知识注入阶段这里就计算出L_{inj,i}的值。
def knowledge_injection_loss(messenger, local_model, data_x, data_y, lambda_l, lambda_m):
    """
    知识注入函数，将 Messenger 模型与 Local 模型的输出通过 Receiver 进行融合
    :param messenger: Messenger 模型实例
    :param local_model: Local 模型实例
    :param data_x: x 数据
    :param data_y: y 真实标签
    :
    :return: L_{inj,i}
    """
    # 步骤 1: 通过 Messenger 计算Messenger的预测值
    local_body_feature = local_model.forward_body(data_x)  # 计算local的body部分的输出

    messenger_preds = messenger.forward(data_x, local_body_feature, "injection")
    # 步骤 2: 通过 Local 模型 计算local的预测值

    local_preds = local_model.forward(data_x).to(device)
    # 计算local知识注入阶段加权的损失值
    total_loss, first_loss, second_loss = local_model.knowledge_injection_loss(messenger_preds, local_preds, data_y,
                                                                               lambda_l, lambda_m)
    return total_loss, first_loss, second_loss


# 知识蒸馏阶段的损失值计算，计算出L_{dis,i}
def knowledge_distillation_loss(messenger, local_model, data_x, data_y, lambda_con, lambda_m):
    """
    知识注入函数，将 Messenger 模型与 Local 模型的输出通过 Receiver 进行融合
    :param messenger: Messenger 模型实例
    :param local_model: Local 模型实例
    :param data_x: x 数据
    :param data_y: y 真实标签
    :return: L_{inj,i}
    """
    # 1. 计算messenger的预测
    local_body_feature = local_model.forward_body(data_x)  # 计算local的body部分的输出
    messenger_preds = messenger.forward(data_x, local_body_feature, "distillation")

    # 2. 计算local 的预测
    local_preds = local_model.forward(data_x)

    # 3. 传递给messenger的知识蒸馏阶段损失函数计算损失值
    total_loss, first_loss, second_loss = messenger.knowledge_distillation_loss(local_preds, messenger_preds, data_y,
                                                                                lambda_m, lambda_con)
    return total_loss, first_loss, second_loss


# 知识注入阶段
def knowledge_injection_stage(messengers, local_models, dataloaders, client_nums, num_epochs=4, lambda_l=0.9,
                              lambda_m=0.1, lr=0.0001):
    """
    知识注入阶段的训练函数
    :param messengers: Messenger 模型实例
    :param local_models: 本地模型实例
    :param receivers: 接收器模型实例，用于融合特征
    :param dataloaders: 数据加载器，提供训练数据
    :param num_epochs: 本地模型训练的轮次
    :param lambda_l: 本地模型损失权重
    :param lambda_m: 信使模型损失权重
    """

    # 确保 messengers 和 local_models 的数量等于 client_nums
    assert len(messengers) == client_nums, "messengers 的数量应与 client_nums 一致"
    assert len(local_models) == client_nums, "local_models 的数量应与 client_nums 一致"

    # 为每个客户端的 local_model 定义优化器
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in local_models]  # 学习率为 0.0001
    # 为每个reciver定义优化器，利用梯度更新参数
    optimizers_receiver = [optim.Adam(model.receiver.parameters(), lr=receiver_lr) for model in
                           messengers]  # 学习率为 0.0001

    losses = []  # 记录所有客户端的loss
    accuracies = []  # 记录所有客户端的accuracy

    for i in range(client_nums):
        local_model = local_models[i]  # 当前客户端的本地模型
        messenger = messengers[i]  # 当前客户端的信使模型
        optimizer = optimizers[i]  # 当前客户端的优化器
        optimizer_receiver = optimizers_receiver[i]  # 当前客户端对应的receiver的优化器
        dataloader = dataloaders[i]  # 当前客户端的训练数据
        local_model = local_model.to(device)  # 将模型放到GPU上
        messenger = messenger.to(device)  # 将messenger模型放到GPU上

        # 冻结梯度：仅在知识注入阶段冻结 Messenger 模型的梯度
        # 冻结 Messenger 的所有参数
        for param in messenger.parameters():
            param.requires_grad = False
        # 不冻结LocalModel
        for param in local_model.parameters():
            param.requires_grad = True

        # 设置模型为训练模式
        local_model.train()

        client_losses = []  # 当前客户端每个 epoch 的 loss
        client_accuracies = []  # 当前客户端每个 epoch 的 accuracy

        for epoch in range(num_epochs):  # 进行 num_epochs 轮本地训练
            # print(f"local_model{i+1}: Epoch [{epoch + 1}/{num_epochs}]")
            # 训练模型
            total_loss = 0.0
            total_first_loss = 0.0
            total_second_loss = 0.0
            correct_preds = 0
            total_samples = 0
            for data_x, data_y in dataloader:
                data_x, data_y = data_x.to(device), data_y.to(device)  # 将数据移动到 GPU

                # 清空梯度
                optimizer.zero_grad()
                # 计算知识注入损失
                loss, first_loss, second_loss = knowledge_injection_loss(messenger, local_model, data_x, data_y,
                                                                         lambda_l,
                                                                         lambda_m)
                # 反向传播
                loss.backward()
                # 设置梯度裁剪
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(messenger.receiver.parameters(), max_norm=1.0)

                # 更新本地模型参数
                optimizer.step()
                # 更新receiver的参数
                optimizer_receiver.step()

                # 统计 loss 和 accuracy
                total_loss += loss.item() * data_x.size(0)  # 累加每个batch的loss
                total_first_loss += first_loss.item() * data_x.size(0)
                total_second_loss += second_loss.item() * data_x.size(0)
                _, preds = torch.max(local_model(data_x), 1)  # 获取预测值
                correct_preds += (preds == data_y).sum().item()  # 累加正确预测的数量
                total_samples += data_y.size(0)  # 累加样本数量

            # 计算每个 epoch 的平均 loss 和 accuracy
            epoch_loss = total_loss / total_samples
            epoch_accuracy = correct_preds / total_samples
            epoch_first_loss = total_first_loss / total_samples
            epoch_second_loss = total_second_loss / total_samples
            client_losses.append(epoch_loss)
            client_accuracies.append(epoch_accuracy)

            print(
                f"Client {i + 1}, Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, first_loss: {epoch_first_loss:.4f}, second_loss: {epoch_second_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        losses.append(client_losses)
        accuracies.append(client_accuracies)

        local_model.eval()  # 调整local模型为评估模式
        # 恢复梯度
        for param in messenger.parameters():
            param.requires_grad = True
        for param in local_model.parameters():
            param.requires_grad = True
    print("Knowledge Injection Phase Completed")
    return losses, accuracies  # 返回各个客户端的losses和accuracies


# 知识蒸馏阶段
def knowledge_distillation_stage(messengers, local_models, data_loaders, client_nums, lambda_con=0.1, lambda_m=0.9,
                                 lr=0.00001, num_epochs=1):
    """
    知识蒸馏阶段的训练，冻结本地模型，仅更新每个客户端的信使模型。

    :param messengers: 信使模型列表，每个元素为 Messenger 模型实例
    :param local_models: 本地模型列表，每个元素为 Local 模型实例
    :param data_loaders: 数据加载器列表，每个元素为 DataLoader 实例
    :param client_nums: 客户端数量
    :param lambda_con: KL 散度损失的权重
    :param lambda_m: 信使交叉熵损失的权重
    :param lr: 学习率
    :param num_epochs: 训练的轮数，默认为 1 轮
    """

    # 为每个客户端的 messenger 定义优化器
    optimizers = [optim.Adam(messengers[i].parameters(), lr=lr) for i in range(client_nums)]
    # 为每个transmitter定义优化器
    optimizers_transmitter = [optim.Adam(model.transmitter.parameters(), lr=transmitter_lr) for model in
                              messengers]  # 学习率为 0.0001
    # 每个客户端独立训练其信使模型
    for i in range(client_nums):
        messenger = messengers[i]  # 获取当前客户端的信使模型
        local_model = local_models[i]  # 获取当前客户端的本地模型
        data_loader = data_loaders[i]  # 获取当前客户端的数据加载器
        optimizer = optimizers[i]  # 获取当前客户端的优化器

        optimizer_transmitter = optimizers_transmitter[i]  # 获取当前的transmitter
        # 将模型放到GPU上
        local_model = local_model.to(device)
        messenger = messenger.to(device)
        messenger.train()  # 设置信使模型为训练模式

        # 冻结梯度：仅在知识蒸馏阶段冻结 LocalModel 模型的梯度
        # 冻结 LocalModel 的所有参数
        for param in messenger.parameters():
            param.requires_grad = True
        # 不冻结LocalModel
        for param in local_model.parameters():
            param.requires_grad = False
        # 训练循环
        for epoch in range(num_epochs):
            # print(f"local_model {i+1}: Epoch [{epoch + 1}/{num_epochs}]")
            total_loss = 0.0
            total_first_loss = 0.0
            total_second_loss = 0.0
            total_samples = 0
            for data_x, data_y in data_loader:
                data_x, data_y = data_x.to(device), data_y.to(device)  # 将数据放到模型上

                # 清空梯度
                optimizer.zero_grad()
                # 执行知识蒸馏训练步骤
                total_loss, first_loss, second_loss = knowledge_distillation_loss(
                    messenger,
                    local_model,
                    data_x,
                    data_y,
                    lambda_con,
                    lambda_m
                )
                # 反向传播和优化
                total_loss.backward()
                # 设置梯度裁剪
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(messenger.transmitter.parameters(), max_norm=1.0)
                optimizer.step()
                # 更新transmitter的参数
                optimizer_transmitter.step()

                # 统计 loss 和 accuracy
                total_loss += total_loss.item() * data_x.size(0)  # 累加每个batch的loss
                total_first_loss += first_loss.item() * data_x.size(0)
                total_second_loss += second_loss.item() * data_x.size(0)
                total_samples += data_y.size(0)  # 累加样本数量
            # 计算每个 epoch 的平均 loss 和 accuracy
            epoch_loss = total_loss / total_samples
            epoch_first_loss = total_first_loss / total_samples
            epoch_second_loss = total_second_loss / total_samples
            print(
                f"Client {i + 1}, Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, first_loss: {epoch_first_loss:.4f}, second_loss: {epoch_second_loss:.4f}")
        messenger.eval()  # 设置信使模型为评估模式

        # 恢复梯度
        for param in messenger.parameters():
            param.requires_grad = True
        for param in local_model.parameters():
            param.requires_grad = True
    print("Knowledge Distillation Phase Completed")


# 信使模型的上传以及聚合操作
def messenger_uploadAndAggregated(server, messengers, weights=None):
    """
    接收多个客户端的信使模型参数，进行加权聚合，并将聚合后的参数更新到服务器模型。

    :param server: 服务器端的模型实例（Server 类）。
    :param messengers: 一个包含多个客户端 Messenger 模型的列表。
    :param weights: 一个列表，包含每个客户端的权重。若未提供，则默认为均等权重。
    """
    # 1. 上传客户端的模型（body 和 head 部分参数）到服务器
    server.receive_client_models_params(messengers)

    # 2. 聚合所有客户端的信使模型的 body 部分
    server.aggregate_body(messengers, weights)

    # 3. 聚合所有客户端的信使模型的 head 部分
    server.aggregate_head(messengers, weights)

    # 4. 更新服务器的模型参数（body 和 head）
    # 将聚合后的参数更新到服务器端的模型
    server.update_body(messengers)
    server.update_head(messengers)

    # 打印信息表示聚合和更新已完成
    print("Messenger model parameters have been uploaded and aggregated successfully.")


# 更新信使模型的body部分,传递信使模型列表
def messenger_body_update(server, messengers):
    server.update_body(messengers)
    print("Messenger model's body parameters have been updated successfully")


# 更新信使模型的head部分,传递信使模型列表
def messenger_head_update(server, messengers):
    server.update_head(messengers)
    print("Messenger model's head parameters have been updated successfully")


"""
    实现ResNet{17、11、8、5}
    17这些数字由卷积层 + 全连接层个数确定。
"""


# 定义 Bottleneck 残差块,残差块包含1x1卷积层 + 3x3卷积层 + 1x1卷积层
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        # 第一个 1x1 卷积层，用于调整通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # 对应的批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个 3x3 卷积层，用于特征提取
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 对应的批归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 第三个 1x1 卷积层，恢复原始的通道数
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # 对应的批归一化层
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 当输入和输出的尺寸或通道数不同时，添加降采样层
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # 使用 1x1 卷积调整尺寸和通道数
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 保存输入值用于残差连接
        identity = x
        # 如果需要降采样，先调整输入
        if self.downsample is not None:
            identity = self.downsample(x)

        # 第一个 1x1 卷积层和批归一化层，激活函数为 ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个 3x3 卷积层和批归一化层，激活函数为 ReLU
        out = F.relu(self.bn2(self.conv2(out)))
        # 第三个 1x1 卷积层和批归一化层
        out = self.bn3(self.conv3(out))
        # 将输入 (identity) 加到输出 (out) 上，实现残差连接
        out += identity
        # ReLU 激活
        return F.relu(out)


# 定义 ResNet 模型
class ResNet(nn.Module):
    def __init__(self, layers, num_classes=2):
        super(ResNet, self).__init__()

        # 初始 3x3 卷积层，输出通道数为 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始批归一化层
        self.bn1 = nn.BatchNorm2d(64)

        # 构建四个阶段的残差块层，每个阶段的层数由 layers 参数控制
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1)  # 第一阶段
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2) if layers[
                                                                            1] > 0 else None  # 如果layers[1]为0，则不创建这么一个层
        # 如果 layers 的长度超过 2，构建第三阶段
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2) if layers[2] > 0 else None
        # 如果 layers 的长度超过 3，构建第四阶段
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2) if layers[3] > 0 else None

        # 全局平均池化层，输出大小为 (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 计算实际的最后一层输出通道数
        final_channels = 64  # 最后的输出的通道数
        if self.layer2:
            final_channels = 128
        if self.layer3:
            final_channels = 256
        if self.layer4:
            final_channels = 512

        self.final_channels = final_channels  # 存储body的输出最后通道数
        # print(f"final_channels: {final_channels}")
        # 全连接层，将最后一层的输出映射到类别数
        self.fc = nn.Linear(final_channels, num_classes)

        # 动态构建 body 和 head
        self.body = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            *(layer for layer in [self.layer2, self.layer3, self.layer4] if layer is not None),
            self.avg_pool
        )

        # 定义 head 部分（展平层和全连接层），并且softmax
        self.head = nn.Sequential(
            nn.Flatten(),
            self.fc,
            nn.Softmax(dim=1)  # 对类别维度做 softmax
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        # 创建一个由多个 BottleneckBlock 组成的层
        layers = []
        # 第一个残差块，可能涉及下采样
        layers.append(BottleneckBlock(in_channels, out_channels, stride))
        # 后续的残差块，不涉及下采样
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_channels, out_channels))
        # 返回由多个残差块组成的层
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        # 输入通过初始卷积层和批归一化层
        out = F.relu(self.bn1(self.conv1(x)))
        # 依次通过各阶段的残差块层
        out = self.layer1(out)
        # 如果第二阶段存在残差块层。
        if self.layer2:
            out = self.layer2(out)
        # 如果存在第三阶段，继续通过
        if self.layer3:
            out = self.layer3(out)
        # 如果存在第四阶段，继续通过
        if self.layer4:
            out = self.layer4(out)
        # 全局平均池化，将特征图缩小为 (1, 1)
        out = self.avg_pool(out)

        # 展平特征图，用于全连接层
        out = torch.flatten(out, 1)
        # 最后通过全连接层输出分类结果
        out = self.fc(out)
        """
        # 通过 body 部分
        out = self.body(x)
        # 通过 head 部分
        out = self.head(out)
        return out

    # 返回body部分输出的最后通道数
    def get_body_output_channels_num(self):
        return self.final_channels


# 定义各 ResNet 变体的构建函数
def resnet17(num_classes=2):
    # 返回 ResNet-17 实例
    return ResNet([1, 1, 2, 1], num_classes)


def resnet11(num_classes=2):
    # 返回 ResNet-11 实例
    return ResNet([1, 1, 1, 0], num_classes)


def resnet8(num_classes=2):
    # 返回 ResNet-8 实例
    return ResNet([1, 1, 0, 0], num_classes)


def resnet5(num_classes=2):
    # 返回 ResNet-5 实例
    return ResNet([1, 0, 0, 0], num_classes)


# 这个类用于生成dataset,应该没bug
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        自定义 Dataset 类，用于封装图像和标签数据

        :param images: 图像列表
        :param labels: 标签列表
        :param transform: 可选的图像转换（如归一化）
        """
        self.images = images
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 统一尺寸为 224x224
            transforms.ToTensor()  # 转换为 Tensor
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 如果有定义转换，则对图像进行转换
        if self.transform:
            image = self.transform(image)

        # 检查 image 类型，如果不是 Tensor 才调用 ToTensor()
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        return image, label


# 两个特征图像的匹配
def match_dimensions(x, local_body_feature):
    # 获取 x 和 local_body_feature 的空间维度
    h1, w1 = x.shape[2:]  # x 的高度和宽度
    h2, w2 = local_body_feature.shape[2:]  # local_body_feature 的高度和宽度

    # 如果 x 的空间维度大于 local_body_feature，则填充 local_body_feature
    if h1 > h2 or w1 > w2:
        # 计算需要填充的高度和宽度
        pad_h = h1 - h2
        pad_w = w1 - w2
        # 计算填充的上下左右
        local_body_feature = F.pad(local_body_feature, (0, pad_w, 0, pad_h))  # 左右填充 (w)，上下填充 (h)

    # 如果 x 的空间维度小于 local_body_feature，则裁剪 local_body_feature
    elif h1 < h2 or w1 < w2:
        # 裁剪 local_body_feature 使其与 x 的空间维度一致
        local_body_feature = local_body_feature[:, :, :h1, :w1]

    return local_body_feature


# 随机划分数据集
def split_dataset(dataset, train_ratio=0.7, random_seed=17):
    """
    按照给定比例划分数据集为训练集和测试集。

    :param dataset: 输入的完整数据集
    :param train_ratio: 训练集占比，默认为 0.7
    :param random_seed: 控制划分
    :return: 划分后的训练集和测试集
    """

    # 设置随机种子以确保划分的一致性
    torch.manual_seed(random_seed)

    # 计算训练集和测试集的大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # 使用 random_split 划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


# 加载数据,根据文件夹的路径，加载其中放大倍数对应的文件夹的所有图片,或者不传递放大倍数，则所有图片都读取
def dataReader(folder_path, magnification=None, batch_size=8, scale=1):
    """
    读取指定文件夹路径下所有特定放大倍数（如 40X）的图片，并返回图片列表和图片数量。

    :param folder_path: 包含图片的文件夹路径
    :param magnification: 需要加载的图片放大倍数（默认为"40X"）
    :return: 加载的图片列表和图片数量
    """
    images = []  # 用于存储加载的图片
    images_y = []  # 用于存储图像的标签（0表示良性B，1表示恶性M）
    count_images = 0  # 用于统计加载的图片数量

    # 检查给定的文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("提供的路径不存在！")
        return [], [], 0  # 路径不存在时，返回空列表和0数量

        # 定义支持的图片扩展名（包含大小写）
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        # print(f"正在处理目录: {root}")  # 调试用，打印当前目录
        # 只选择包含指定放大倍数的文件夹
        if magnification is not None and magnification not in root:  # 如果没有指定放大倍数，或者路径中包含指定的放大倍数
            continue  # 如果指定放大倍数，且该倍数文件夹不在该路径则跳过
        for filename in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, filename)

            # 检查是否为支持的图片文件
            if filename.lower().endswith(valid_extensions):  # 忽略大小写匹配扩展名
                try:
                    # 打开图片并转换为 RGB 模式
                    img = Image.open(file_path).convert('RGB')
                    images.append(img)
                    count_images += 1

                    # 提取肿瘤类型标签
                    file_name = os.path.splitext(filename)[0]  # 去掉扩展名
                    if '_' in file_name:
                        tumor_type = file_name.split('_')[1]
                        if tumor_type == "B":
                            images_y.append(0)  # 良性
                        elif tumor_type == "M":
                            images_y.append(1)  # 恶性
                        else:
                            print(f"无法识别肿瘤类型: {tumor_type}，跳过该图片")
                    else:
                        print(f"文件名格式错误: {filename}，跳过该图片")

                except Exception as e:
                    print(f"无法加载图片 {file_path}，错误: {e}")

    # 创建 Dataset 对象
    dataset = CustomDataset(images, images_y)
    # 划分数据集
    train_dataset, test_dataset = split_dataset(dataset, train_ratio, random_seed)
    # 如果是None代表读取所有图片，要进行下采样
    if magnification is None:
        train_dataset, test_dataset, train_img_nums, test_img_nums = generate_downsampled_datasets(train_dataset,
                                                                                                   test_dataset, scale)
        # 创建 DataLoader 对象
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                                       drop_last=True)  # 丢弃最后批次不足batch_size的情况，避免batch_size=1的情况
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
        return train_dataloader, test_dataloader, train_img_nums + test_img_nums  # 返回包含所有加载图片的列表、标签和图片数量

    # 创建 DataLoader 对象
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                                   drop_last=True)  # 丢弃最后批次不足batch_size的情况，避免batch_size=1的情况
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, test_dataloader, count_images  # 返回包含所有加载图片的列表、标签和图片数量


# 产生下采样的函数
def generate_downsampled_datasets(train_dataset, test_dataset, scale=1):
    """
    对传入的训练和测试 dataset 进行指定的下采样倍率（如 1, 2, 4, 8），
    下采样后恢复到原始大小，返回处理后的 datasets 及图像数量。

    :param train_dataset: 训练数据集
    :param test_dataset: 测试数据集
    :param scale: 下采样倍率（如 2, 4, 8 等）
    :return: 下采样后的训练和测试 datasets 及图像数量
    """

    # 定义内部下采样函数
    def downsample_and_restore(image, scale_factor):
        """
        对单张图像进行下采样（缩小）并恢复原始大小。

        :param image: 输入图像（Tensor 格式）
        :param scale_factor: 下采样倍率
        :return: 下采样后的图像（保留原始大小）
        """
        # 获取原始尺寸
        original_height = image.size(1)
        original_width = image.size(2)

        # 缩小图像
        reduced_height = original_height // scale_factor
        reduced_width = original_width // scale_factor
        reduced_image = transforms.functional.resize(image, [reduced_height, reduced_width])

        # 恢复到原始大小
        restored_image = transforms.functional.resize(reduced_image, [original_height, original_width])
        return restored_image

    # 初始化训练和测试数据的图像和标签列表
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # 处理训练数据集
    for image, label in train_dataset:
        downsampled_image = downsample_and_restore(image, scale)
        train_images.append(downsampled_image)
        train_labels.append(torch.tensor(label))  # 确保标签是张量

    # 处理测试数据集
    for image, label in test_dataset:
        downsampled_image = downsample_and_restore(image, scale)
        test_images.append(downsampled_image)
        test_labels.append(torch.tensor(label))  # 确保标签是张量

    # 合并成 TensorDataset
    train_dataset_downsampled = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
    test_dataset_downsampled = TensorDataset(torch.stack(test_images), torch.stack(test_labels))

    # 返回处理后的数据集和图像数量
    return train_dataset_downsampled, test_dataset_downsampled, len(train_images), len(test_images)


# 计算模型的ACC和MF1
def evaluate_model(model, test_dataloader):
    """
    评估模型性能，计算准确率和宏平均F1得分。

    参数:
    - model: 需要评估的模型。
    - dataloader: 用于测试的dataloader。

    返回:
    - acc: 准确率。
    - mf1: 宏平均F1得分。
    """
    model.eval()  # 将模型设置为评估模式，以禁用dropout等训练时特有的行为
    all_preds = []  # 用于存储所有批次的预测结果
    all_labels = []  # 用于存储所有批次的真实标签

    with torch.no_grad():  # 禁用梯度计算，以提高推理速度并节省内存
        for inputs, labels in test_dataloader:  # 逐批处理test_dataloader中的数据
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签放到GPU设备上
            outputs = model(inputs)  # 前向传播，计算模型的输出
            _, preds = torch.max(outputs, 1)  # 获取每个输入的预测类别（输出值最大的类别）

            # 将预测结果和真实标签从GPU转到CPU，并转换为numpy格式，便于后续计算
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 使用sklearn计算准确率（acc）和宏平均F1得分（mf1）
    acc = accuracy_score(all_labels, all_preds)  # 计算准确率
    mf1 = f1_score(all_labels, all_preds, average='macro')  # 计算宏平均F1得分

    return acc, mf1  # 返回准确率和宏平均F1得分


def correspondence(server, messenger_models, local_models, train_dataloaders, test_dataloaders):
    # 初始化

    weights = None  # 权重
    clients_num = 4  # 客户端的个数
    tag = True
    # 获取第一个 train_dataloader 的第一个 batch,用于初始化messenger的transmitter和receiver
    data_x, data_y = next(iter(train_dataloaders[0]))
    data_x, data_y = data_x.to(device), data_y.to(device)  # 将数据移动到 GPU
    for i in range(len(messenger_models)):
        I_mes = messenger_models[i].body(data_x)
        I_loc = local_models[i].forward_body(data_x)
        messenger_models[i].receiver(I_mes, I_loc)
        messenger_models[i].transmitter(I_loc, I_mes)

    # 用于存储最后一轮各个客户端的性能指标
    all_metrics = []

    # 保存各客户端在每轮通信中的loss和accuracy
    client_losses = [[] for _ in range(clients_num)]
    client_accuracies = [[] for _ in range(clients_num)]

    # 通信轮次为num_rounds
    for round_num in range(num_rounds):
        print(f"第{round_num + 1}轮通信：")

        # 知识注入阶段
        if (tag):
            print("开始知识注入阶段")
        losses, accuracies = knowledge_injection_stage(messenger_models, local_models, train_dataloaders, clients_num,
                                                       num_epochs_inj, lambda_l, lambda_m, lr_injection)
        # 将每个客户端的 loss 和 accuracy 保存
        for i in range(clients_num):
            client_losses[i].extend(losses[i])
            client_accuracies[i].extend(accuracies[i])

        if (tag):
            print("开始知识蒸馏阶段")
        # 知识蒸馏阶段
        knowledge_distillation_stage(messenger_models, local_models, train_dataloaders, clients_num, lambda_con,
                                     lambda_m_dis,
                                     lr_distillation, num_epochs_dis)
        # 信使模型上传到服务器端进行聚合
        messenger_uploadAndAggregated(server, messenger_models, weights)
        if ifUpdateMesBody == True:
            # 更新信使模型的body部分
            messenger_body_update(server, messenger_models)
        if ifUpdateMesHead == True:
            # 更新信使模型的head部分
            messenger_head_update(server, messenger_models)


    # 通信结束保存模型
    # 确保目标目录存在
    os.makedirs("models", exist_ok=True)
    for i in range(len(local_models)):
        torch.save(local_models[i].state_dict(),
                   f"models\\client_{i + 1}_round_{round_num + 1}_{lambda_l}_{lambda_m}_{lambda_con}_{lambda_m_dis}_all.pt")
    print(f"进行了{num_rounds}次通信")


# 医学图像分类任务
def medical_image_classification(train_dataloaders, test_dataloaders, server, local_models, messenger_models):
    # 将每个 local_model 移动到设备
    for i in range(len(local_models)):
        local_models[i] = local_models[i].to(device)
    # 将每个 messenger_model 移动到设备
    for i in range(len(messenger_models)):
        messenger_models[i] = messenger_models[i].to(device)
    server = server.to(device)
    print("开始通信")
    correspondence(server, messenger_models, local_models, train_dataloaders, test_dataloaders)


# 医学图像分类任务仅仅本地训练
def medical_image_classification_only_local(train_dataloaders, test_dataloaders, local_models):
    local_model_resnet17 = local_models[0]
    local_model_resnet11 = local_models[1]
    local_model_resnet8 = local_models[2]
    local_model_resnet5 = local_models[3]
    # 创建用于存储指标的文件
    output_file = "only_local_train_model_metrics.csv"
    epochs = onlyLocal_epoch  # 模型的本地训练次数
    # 打开文件写入
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["Model", "Accuracy", "MF1"])

        # # ResNet17 训练和测试
        print("ResNet17 开始训练")
        local_model_resnet17.train_model(train_dataloaders[0], epochs)
        print("ResNet17 开始测试")
        acc_17, mf1_17 = evaluate_model(local_model_resnet17, test_dataloaders[0])
        print(f"ResNet17 - Accuracy: {acc_17:.4f}, MF1: {mf1_17:.4f}")
        # 将结果保存到文件
        writer.writerow(["ResNet17", acc_17, mf1_17])

        # ResNet11 训练和测试
        print("ResNet11 开始训练")
        local_model_resnet11.train_model(train_dataloaders[1], epochs)
        print("ResNet11 开始测试")
        acc_11, mf1_11 = evaluate_model(local_model_resnet11, test_dataloaders[1])
        print(f"ResNet11 - Accuracy: {acc_11:.4f}, MF1: {mf1_11:.4f}")
        writer.writerow(["ResNet11", acc_11, mf1_11])

        # ResNet8 训练和测试
        print("ResNet8 开始训练")
        local_model_resnet8.train_model(train_dataloaders[2], epochs)
        print("ResNet8 开始测试")
        acc_8, mf1_8 = evaluate_model(local_model_resnet8, test_dataloaders[2])
        print(f"ResNet8 - Accuracy: {acc_8:.4f}, MF1: {mf1_8:.4f}")
        writer.writerow(["ResNet8", acc_8, mf1_8])

        # ResNet5 训练和测试
        print("ResNet5 开始训练")
        local_model_resnet5.train_model(train_dataloaders[3], epochs)
        print("ResNet5 开始测试")
        acc_5, mf1_5 = evaluate_model(local_model_resnet5, test_dataloaders[3])
        print(f"ResNet5 - Accuracy: {acc_5:.4f}, MF1: {mf1_5:.4f}")
        writer.writerow(["ResNet5", acc_5, mf1_5])

    print(f"所有模型的评估结果已保存到 '{output_file}' 文件中。")


# 保存所有通信轮次的评估指标到文件
def save_metrics_to_file(metrics, filename="metrics.csv"):
    """
    将每个通信轮次的准确率和宏平均F1值保存到文件。

    :param metrics: 存储每个通信轮次的每个客户端的性能指标 (acc, mf1)
    :param filename: 保存的文件名，默认为 "metrics.csv"
    """
    # 打开文件进行写入，如果文件不存在则会创建
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(['Round', 'Metrics'])

        # 遍历每个通信轮次，写入每个客户端的acc和mf1
        for round_num, round_metrics in enumerate(metrics, start=1):
            # round_metrics 是一个列表，包含每个客户端的 (acc, mf1)
            metrics_str = ', '.join([f'acc: {acc:.4f}, mf1: {mf1:.4f}' for acc, mf1 in round_metrics])
            row = [f'R: {round_num}', metrics_str]
            writer.writerow(row)


# 医学图像分类任务的泛化实验
"""
对所有的另外3个分别进行泛化
"""


# 绘制通信结束后各个客户端loss以及acc曲线图
def plot_loss_and_accuracy(client_losses, client_accuracies, num_rounds):
    """
    绘制每个客户端的loss和accuracy曲线，分别保存成两张图片。
    """
    # 绘制 Loss 曲线
    plt.figure(figsize=(12, 6))
    for i, losses in enumerate(client_losses):
        plt.plot(range(1, len(losses) + 1), losses, label=f'Client {i + 1}', linestyle='-')
    plt.title('Loss per Epoch for Each Client')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.suptitle(f'Training Loss Across {num_rounds} Rounds')
    plt.tight_layout()
    plt.savefig("client_loss_curve.png")
    plt.show(block=False)
    plt.pause(5)  # 展示5秒后关闭
    plt.close()  # 关闭图像

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(12, 6))
    for i, accuracies in enumerate(client_accuracies):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'Client {i + 1}', linestyle='--')
    plt.title('Accuracy per Epoch for Each Client')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle(f'Training Accuracy Across {num_rounds} Rounds')
    plt.tight_layout()
    plt.savefig("client_accuracy_curve.png")
    plt.show(block=False)
    plt.pause(5)  # 展示5秒后关闭
    plt.close()  # 关闭图像


"""


初始化变量

"""


# 初始化函数,读取数据，实例化server、local、messenger
def init():
    # 读取数据 按照40X、100X、200X、400X进行读取
    # train_dataloader_1, test_dataloader_1, count_images_1 = dataReader(folder_path, magnification_400, batch_size)
    # train_dataloader_2, test_dataloader_2, count_images_2 = dataReader(folder_path, magnification_200, batch_size)
    # train_dataloader_3, test_dataloader_3, count_images_3 = dataReader(folder_path, magnification_100, batch_size)
    # train_dataloader_4, test_dataloader_4, count_images_4 = dataReader(folder_path, magnification_40, batch_size)

    # 读取数据 按照下采样读取
    # train_dataloader_1, test_dataloader_1, count_images_1 = dataReader(folder_path, None, batch_size,1)     # 原始分辨率
    # train_dataloader_2, test_dataloader_2, count_images_2 = dataReader(folder_path, None, batch_size, 2)    # 2倍缩小
    # train_dataloader_3, test_dataloader_3, count_images_3 = dataReader(folder_path, None, batch_size, 4)    # 4倍缩小
    # train_dataloader_4, test_dataloader_4, count_images_4 = dataReader(folder_path, None, batch_size, 8)    # 8倍缩小
    print(f"client1_采样倍数为1,随机数种子为{random_seed}的数据集:")
    train_dataloader_1, test_dataloader_1, count_images_1 = readData(folder_path, batch_size, train_ratio, random_seed,
                                                                     1)
    print(f"client2_采样倍数为2,随机数种子为{random_seed + 10}的数据集:")
    train_dataloader_2, test_dataloader_2, count_images_2 = readData(folder_path, batch_size, train_ratio,
                                                                     random_seed + 10, 2)
    print(f"client3_采样倍数为4,随机数种子为{random_seed + 100}的数据集:")
    train_dataloader_3, test_dataloader_3, count_images_3 = readData(folder_path, batch_size, train_ratio,
                                                                     random_seed + 100, 4)
    print(f"client4_采样倍数为8,随机数种子为{random_seed + 1000}的数据集:")
    train_dataloader_4, test_dataloader_4, count_images_4 = readData(folder_path, batch_size, train_ratio,
                                                                     random_seed + 1000, 8)

    print("读取数据完成")
    print(
        f"图片数量:\nclient_1:{count_images_1} \nclient_2:{count_images_2} \nclient_3:{count_images_3} \nclient_4:{count_images_4}")
    train_dataloaders = [train_dataloader_1, train_dataloader_2, train_dataloader_3,
                         train_dataloader_4]  # 训练数据加载器集，对应不同客户端的本地训练数据集
    test_dataloaders = [test_dataloader_1, test_dataloader_2, test_dataloader_3,
                        test_dataloader_4]  # 测试数据加载器集，对应不同客户端的测试训练数据集
    # 并且将数据加载进去.
    local_model_resnet17 = LocalModel('resnet17', num_classes, train_dataloader_1)
    local_model_resnet11 = LocalModel('resnet11', num_classes, train_dataloader_2)
    local_model_resnet8 = LocalModel('resnet8', num_classes, train_dataloader_3)
    local_model_resnet5 = LocalModel('resnet5', num_classes, train_dataloader_4)
    print("加载数据到local_model完成")
    # 创建4个messenger模型
    messenger_resnet17 = Messenger(input_size_mes, local_model_resnet17.body_output_channels, output_size,
                                   num_classes=2)
    messenger_resnet11 = Messenger(input_size_mes, local_model_resnet11.body_output_channels, output_size,
                                   num_classes=2)
    messenger_resnet8 = Messenger(input_size_mes, local_model_resnet8.body_output_channels, output_size, num_classes=2)
    messenger_resnet5 = Messenger(input_size_mes, local_model_resnet5.body_output_channels, output_size, num_classes=2)
    # 实例化1个Server
    server = Server(num_classes=2, num_clients=4)
    local_models = [local_model_resnet17, local_model_resnet11, local_model_resnet8, local_model_resnet5]
    messenger_models = [messenger_resnet17, messenger_resnet11, messenger_resnet8, messenger_resnet5]

    # 将每个 local_model 移动到设备
    for i in range(len(local_models)):
        local_models[i] = local_models[i].to(device)

    # 将每个 messenger_model 移动到设备
    for i in range(len(messenger_models)):
        messenger_models[i] = messenger_models[i].to(device)
    server = server.to(device)

    return train_dataloaders, test_dataloaders, server, local_models, messenger_models


# 医学图像分类泛化实验
def generalizability_test_medical_classification(local_models, test_dataloaders, generalizability_result_save_path):
    """
    每个客户端的模型对包括自己在内的所有客户端的数据集进行测试，并以矩阵形式保存结果。
    :param local_models: 本地模型列表
    :param test_dataloaders: 测试数据加载器列表
    :param generalizability_result_save_path: 保存结果的文件路径
    """
    num_clients = len(local_models)
    acc_matrix = [[None for _ in range(num_clients)] for _ in range(num_clients)]  # 存储ACC矩阵
    mf1_matrix = [[None for _ in range(num_clients)] for _ in range(num_clients)]  # 存储MF1矩阵

    for i, local_model in enumerate(local_models):
        print(f"正在评估客户端 {i + 1} 的模型...")
        for j, test_dataloader in enumerate(test_dataloaders):
            acc, mf1 = evaluate_model(local_model, test_dataloader)
            print(f"客户端 {i + 1} 的模型在客户端 {j + 1} 的数据集上的表现: ACC={acc:.4f}, MF1={mf1:.4f}")

            # 保存结果到矩阵
            acc_matrix[i][j] = acc
            mf1_matrix[i][j] = mf1

    # 将结果保存为文件（矩阵形式）
    with open(generalizability_result_save_path, 'w') as f:
        # 写入标题
        f.write("Accuracy Matrix:\n")
        f.write("," + ",".join([f"Test_Client_{j + 1}" for j in range(num_clients)]) + "\n")
        for i in range(num_clients):
            f.write(f"Client_{i + 1}," + ",".join([f"{acc_matrix[i][j]:.4f}" for j in range(num_clients)]) + "\n")

        f.write("\nMF1 Matrix:\n")
        f.write("," + ",".join([f"Test_Client_{j + 1}" for j in range(num_clients)]) + "\n")
        for i in range(num_clients):
            f.write(f"Client_{i + 1}," + ",".join([f"{mf1_matrix[i][j]:.4f}" for j in range(num_clients)]) + "\n")

    print(f"所有客户端模型的评估结果已保存为矩阵到 {generalizability_result_save_path}")


# 按照paper划分7：3的随机数据集，返回划分好的路径列表
def split_data(image_files, train_ratio=0.7, random_seed=17):
    """
    将图像文件列表划分为训练集和测试集，确保相同图片的不同分辨率不同时出现在训练集和测试集。
    :param image_files: 包含图像路径的列表
    :param train_ratio: 训练集比例，默认 7:3
    :param random_seed: 随机数种子，确保划分结果可复现
    :return: 训练集和测试集的图像路径列表
    """
    # 按照切片ID分组（将不同放大倍数的图像文件放到同一个切片ID下）
    image_dict = {}
    for image in image_files:
        # 将文件路径按目录层次分割
        parts = image.split('\\')  # 使用反斜杠分割路径
        # 打印路径分割结果进行调试

        slice_id = parts[-3]  # 假设切片ID在路径的倒数第三部分（例如 "SOB_B_A_14-22549AB"）
        magnification = parts[-2].split('X')[0]  # 获取倍数部分（40、100、200、400），通过拆分文件名获取

        # 如果当前切片ID尚未在字典中出现，则添加
        if slice_id not in image_dict:
            image_dict[slice_id] = {'40': [], '100': [], '200': [], '400': []}

        # 将该图像文件路径按放大倍数存储
        image_dict[slice_id][magnification].append(image)

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

        # 用于存储划分后的训练集和测试集
    train_files = []  # 训练集文件路径列表
    test_files = []  # 测试集文件路径列表

    # 首先对所有切片进行随机打乱
    all_images = []
    for slice_id, magnifications in image_dict.items():
        all_images.append((slice_id, magnifications))  # 将切片ID和放大倍数字典一起存储
    random.shuffle(all_images)  # 打乱切片ID的顺序，保证随机性

    # 计算训练集和测试集的数量
    total_images = len(image_files)  # 获取所有图像的数量
    train_size = int(total_images * train_ratio)  # 训练集的目标数量
    test_size = total_images - train_size  # 测试集的目标数量

    # 当前已选择的训练集和测试集数量
    current_train_count = 0  # 记录当前训练集的图像数量
    current_test_count = 0  # 记录当前测试集的图像数量

    # 遍历打乱后的切片信息
    for slice_id, magnifications in all_images:
        # 随机选择每个切片的放大倍数是否属于训练集或测试集
        magnification_choices = list(magnifications.items())  # 获取切片的所有放大倍数及对应的图像
        random.shuffle(magnification_choices)  # 打乱放大倍数的顺序，确保随机性

        # 统计该切片选择的训练集和测试集图像路径
        train_images = []  # 当前切片对应的训练集图像
        test_images = []  # 当前切片对应的测试集图像

        # 遍历每个放大倍数，随机决定是训练集还是测试集
        for magnification, images in magnification_choices:
            if current_train_count < train_size and current_test_count < test_size:
                if current_test_count + current_train_count % 2 == 0:  # 固定条件来决定,这样整个数据集的随机性仅依赖随机种子
                    train_images.append(images)  # 将该倍数的图像添加到训练集
                else:
                    test_images.append(images)  # 将该倍数的图像添加到测试集
            elif current_train_count < train_size:  # 如果训练集数量还未达到目标
                train_images.append(images)  # 将该倍数的图像添加到训练集
            elif current_test_count < test_size:  # 如果测试集数量还未达到目标
                test_images.append(images)  # 将该倍数的图像添加到测试集

        # 若没有划分满足要求则确保训练集和测试集中至少有一个倍数数据，若有一个划分满足要求，则不改变。
        # if len(train_images) == 0 and len(test_images) > 0:
        #     train_images.append(test_images.pop())  # 如果训练集为空，转移一个测试集图像到训练集
        # elif len(test_images) == 0 and len(train_images) > 0:
        #     test_images.append(train_images.pop())  # 如果测试集为空，转移一个训练集图像到测试集

        # 将选择的训练集和测试集图像添加到最终结果中
        for train_image_group in train_images:  # train_images 是一个包含多个子列表的列表
            for image in train_image_group:  # 遍历每个子列表中的图像路径
                train_files.append(image)  # 将训练集图像路径添加到训练集列表
                current_train_count += 1  # 训练集数量加1

        for test_image_group in test_images:  # test_images 是一个包含多个子列表的列表
            for image in test_image_group:  # 遍历每个子列表中的图像路径
                test_files.append(image)  # 将测试集图像路径添加到测试集列表
                current_test_count += 1  # 测试集数量加1

    return train_files, test_files  # 返回划分后的训练集和测试集图像路径


# 处理给定文件夹中的所有图像，并根据规则划分为训练集和测试集，返回train_dataset以及test_dataset
def process_images_from_folder(root_folder, train_ratio=0.7, random_seed=17):
    """
    处理给定文件夹中的所有图像，并根据规则划分为训练集和测试集。
    :param root_folder: 存储图像的根文件夹路径
    :return: 训练集和测试集的图像路径列表
    """
    # 存储所有图像的路径
    image_files = []

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(root_folder):  # 遍历所有目录及文件
        for file in files:  # 遍历每个文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 确保文件是图像（支持png、jpg、jpeg格式）
                # 获取图像的完整路径
                image_path = os.path.join(root, file)  # 通过路径拼接获得完整的图像文件路径
                image_files.append(image_path)  # 将图像路径添加到列表中

    # 调用函数进行数据集划分,返回划分好的路径
    train_files, test_files = split_data(image_files, train_ratio, random_seed)  # 调用 split_data 函数进行划分

    # 根据图像路径读取图像
    train_data = [Image.open(path) for path in train_files]
    test_data = [Image.open(path) for path in test_files]

    # 提取标签（文件名格式为 SOB_B_XXX 或 SOB_M_XXX）
    def extract_labels(file_list):
        labels = []
        for file in file_list:
            file_name = os.path.splitext(os.path.basename(file))[0]  # 获取文件名（不带路径和扩展名）
            parts = file_name.split('_')  # 根据下划线分隔文件名
            if len(parts) > 1:
                tumor_type = parts[1]  # 第二部分是肿瘤类型
                if tumor_type == "B":
                    labels.append(0)  # 良性
                elif tumor_type == "M":
                    labels.append(1)  # 恶性
                else:
                    raise ValueError(f"未知的肿瘤类型: {tumor_type}")
            else:
                raise ValueError(f"文件名格式错误，无法提取标签: {file_name}")
        return labels

    # 提取训练集和测试集的标签
    train_labels = extract_labels(train_files)
    test_labels = extract_labels(test_files)

    # 创建 Dataset
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # 输出训练集和测试集的文件数
    print(f"训练集图像数: {len(train_files)}")  # 输出训练集图像的数量
    print(f"测试集图像数: {len(test_files)}")  # 输出测试集图像的数量

    return train_dataset, test_dataset  # 返回训练集和测试集的 Dataset


# 图片放大倍数
magnification_40 = "40X"
magnification_100 = "100X"
magnification_200 = "200X"
magnification_400 = "400X"
shuffle = True  # 每次加载数据时是否打乱顺序
batch_size = 8
# 一些初始化的数据，比如输入数据的维度，输出数据的维度这些
# input_size_mes = 12800  # 信使body输出的维度 特征加操作时的维度
input_size_mes = 512  # 信使body输出的维度
input_size_loc = 1  # 本地body输出的维度,这个需要动态计算，根据localmodel进行计算
output_size = input_size_mes  # 接收器或传输器输出的维度,也就是head部分的输入维度
num_classes = 2  # 分类任务的类别数
train_ratio = 0.7  # 训练集占0.7
random_seed = 17  # 数据集划分种子
num_rounds = 100  # 通信轮次
num_epochs_inj = 4  # 知识注入轮次
num_epochs_dis = 1  # 知识蒸馏轮次
client_nums = 4  # 客户端个数
lambda_l = 0.9
lambda_m = 0.1
lambda_con = 0.1
lambda_m_dis = 0.9
onlyLocal_epoch = 400  # 仅本地训练的轮次

lr_distillation = 0.00001  # 知识蒸馏阶段学习率
lr_injection = 0.0001  # 知识注入阶段学习率
receiver_lr = 0.0001  # 接收器的学习率
transmitter_lr = 0.0001  # 传输器的学习率
ifUpdateMesHead = True  # 判断是否更新信使头
ifUpdateMesBody = True  # 判断是否更新信使体
isAddFeature = False  # 判断是否仅进行特征加操作

max_loss2 = 100.0  # 自定义阈值

# folder_path = "D:\\ComputerDevFile\\PyCharmProjects\\MH-pFID\\breast_dataset\\breast"
folder_path = "D:\\ComputerDevFile\\PyCharmProjects\\MH-pFID\\test"
only_local_genersalizability_result_save_path = f"only_local_generalizability_result_{lambda_l}_{lambda_m}_{lambda_con}_{lambda_m_dis}.csv"
save_medical_classification_result_filename = f"metrics_{lambda_l}_{lambda_m}_{lambda_con}_{lambda_m_dis}.csv"  # 保存医学分类任务通信后的结果
generalizability_result_save_path = f"generalizability_result_{lambda_l}_{lambda_m}_{lambda_con}_{lambda_m_dis}.csv"
only_local_generalizability_result_save_path = f"only_local_generalizability_result.csv"

total_imgs = 7909  # 各个客户端的数据集的总数
cor = "70"
# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU is available:{torch.cuda.is_available()}")


def main():
    parser = argparse.ArgumentParser(description="命令行参数")

    # 添加多个命令行参数
    parser.add_argument('--path', type=str, help='path是数据的路径', required=True)
    parser.add_argument('--n', type=int, help='client的个数')
    # 解析命令行参数
    args = parser.parse_args()
    return args


# 6. 初始化全局模型并开始训练
if __name__ == "__main__":
    print("main")
    print("测试通信")
    args = main()
    folder_path = args.path  # 将数据路径保存

    train_dataloaders, test_dataloaders, server, local_models, messenger_models = init()
    # medical_image_classification_only_local(train_dataloaders, test_dataloaders, local_models)
    # train_dataloaders, test_dataloaders, server, local_models, messenger_models = init()
    # MH-pFID

    medical_image_classification(train_dataloaders, test_dataloaders, server, local_models, messenger_models)
    # generalizability_test_medical_classification(local_models, test_dataloaders, generalizability_result_save_path)
    # onlyLocal
    # local_models[0].load_model("onlyLocal\\epoch100\\resnet17_only_local.pt")
    # local_models[1].load_model("onlyLocal\\epoch100\\resnet11_only_local.pt")
    # local_models[2].load_model("onlyLocal\\epoch100\\resnet8_only_local.pt")
    # local_models[3].load_model("onlyLocal\\epoch100\\resnet5_only_local.pt")

    # local_models[0].load_model("onlyLocal\\epoch400\\resnet17_only_local.pt")
    # local_models[1].load_model("onlyLocal\\epoch400\\resnet11_only_local.pt")
    # local_models[2].load_model("onlyLocal\\epoch400\\resnet8_only_local.pt")
    # local_models[3].load_model("onlyLocal\\epoch400\\resnet5_only_local.pt")
    # medical_image_classification_only_local(train_dataloaders, test_dataloaders, local_models)
    # generalizability_test_medical_classification(local_models, test_dataloaders, only_local_generalizability_result_save_path)
    generalizability_test_medical_classification(local_models, test_dataloaders, generalizability_result_save_path)
