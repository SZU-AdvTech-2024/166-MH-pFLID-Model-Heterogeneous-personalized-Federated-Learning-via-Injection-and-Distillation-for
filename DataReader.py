
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import os
import random
from PIL import Image                   # 用于将图像分辨率进行下采样

# 这个类用于生成dataset
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

# 按照同一图像不同分辨率作为测试集或训练集划分7：3的随机数据集，返回划分好的路径列表
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
def process_images_from_folder(root_folder,train_ratio=0.7, random_seed=17):
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

# 返回训练dataloader和测试dataloader、以及数据总数量
def readData(root_folder, batch_size=8, train_ratio=0.7, random_seed=17, scale=1):
    train_dataset, test_dataset = process_images_from_folder(root_folder, train_ratio, random_seed)
    # 进行下采样
    train_dataset, test_dataset, train_img_nums, test_img_nums = generate_downsampled_datasets(train_dataset, test_dataset, scale)
    # 创建 DataLoader 对象
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                                   drop_last=True)  # 丢弃最后批次不足batch_size的情况，避免batch_size=1的情况
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader, len(train_dataset)+len(test_dataset)

