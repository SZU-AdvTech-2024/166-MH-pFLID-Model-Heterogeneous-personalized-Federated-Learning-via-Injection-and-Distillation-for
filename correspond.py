import torch
import argparse
from MH_pFLID import Server
from MH_pFLID import LocalModel
from MH_pFLID import Messenger
from DataReader import readData
from MH_pFLID import correspondence
import MH_pFLID

# 初始化函数,读取数据，实例化server、local、messenger
def init():

    print(f"client1_采样倍数为1,随机数种子为{random_seed}的数据集:")
    train_dataloader_1, test_dataloader_1, count_images_1 = readData(folder_path, batch_size, train_ratio, random_seed, 1)
    print(f"client2_采样倍数为2,随机数种子为{random_seed+10}的数据集:")
    train_dataloader_2, test_dataloader_2, count_images_2 = readData(folder_path, batch_size, train_ratio, random_seed+10, 2)
    print(f"client3_采样倍数为4,随机数种子为{random_seed+100}的数据集:")
    train_dataloader_3, test_dataloader_3, count_images_3 = readData(folder_path, batch_size, train_ratio, random_seed+100, 4)
    print(f"client4_采样倍数为8,随机数种子为{random_seed+1000}的数据集:")
    train_dataloader_4, test_dataloader_4, count_images_4 = readData(folder_path, batch_size, train_ratio, random_seed+1000,  8)

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


def main():
    parser = argparse.ArgumentParser(description="命令行参数")

    # 添加多个命令行参数
    parser.add_argument('--path', type=str, help='path是数据的路径')
    parser.add_argument('--r', type=int, help='通信轮次')

    # 解析命令行参数
    args = parser.parse_args()
    return args


"""
    一些全局变量
"""
input_size_mes = 512  # 信使body输出的维度
input_size_loc = 1  # 本地body输出的维度,这个需要动态计算，根据localmodel进行计算
output_size = input_size_mes  # 接收器或传输器输出的维度,也就是head部分的输入维度
num_classes = 2     #分类任务的类别数
train_ratio = 0.7   # 训练集占0.7
batch_size = 8
random_seed = 17    # 数据集划分种子

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU is available:{torch.cuda.is_available()}")

if __name__== "__main__":
    args = main()
    folder_path = args.path  # 将数据路径保存
    MH_pFLID.num_rounds = args.r   # 修改通信轮次
    print(f"开始初始化,通信{MH_pFLID.num_rounds}轮")

    train_dataloaders, test_dataloaders, server, local_models, messenger_models = init()
    medical_image_classification(train_dataloaders, test_dataloaders, server, local_models, messenger_models)   # 进行分类任务训练

