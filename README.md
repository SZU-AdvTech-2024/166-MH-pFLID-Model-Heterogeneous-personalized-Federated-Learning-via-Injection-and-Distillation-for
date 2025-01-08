**训练模型**

python correspond.py --path fold_path_name --r 10

~~~
--path 数据文件路径
--r: 通信轮次
~~~



例如：

~~~
python correspond.py --path test --r 1
~~~

模型将保存在该目录下\models文件夹。

**测试各个客户端在测试集下的训练结果**

~~~
python model_test.py --model_path fold_path_name --d data_path

--path:模型的存储的文件名且，该文件如果是4个客户端则按照下列结构命名
	- client1.pt
	- client2.pt
	- client3.pt
	- client4.pt
--d :数据的路径,需要包含所有数据集，因为会进行划分。


--save_path:模型测试的结果存放路径，可选。

~~~



例如：

~~~
python model_test.py --path models --d breast_dataset\\breast
~~~



