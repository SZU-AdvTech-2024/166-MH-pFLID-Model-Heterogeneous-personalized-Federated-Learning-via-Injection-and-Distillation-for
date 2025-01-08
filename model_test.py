import torch
import argparse
import MH_pFLID
from MH_pFLID import generalizability_test_medical_classification
from MH_pFLID import init
def main():
    parser = argparse.ArgumentParser(description="命令行参数")

    # 添加多个命令行参数
    parser.add_argument('--path', type=str, help='模型的路径')
    parser.add_argument('--d', type=str, help='数据的路径')
    parser.add_argument('--save_path', type=str, help='评估结果的存储路径')

    # 解析命令行参数
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = main()
    models = args.path
    model_test_save_path = MH_pFLID.generalizability_result_save_path
    if args.save_path is not None:
        model_test_save_path = args.save_path
    if args.d is not None:
        MH_pFLID.folder_path = args.d
    train_dataloaders, test_dataloaders, server, local_models, messenger_models = init()
    # 加载模型
    local_models[0].load_model(models + "\\client1.pt")
    local_models[1].load_model(models + "\\client2.pt")
    local_models[2].load_model(models + "\\client3.pt")
    local_models[3].load_model(models + "\\client4.pt")
    generalizability_test_medical_classification(local_models, test_dataloaders, model_test_save_path)