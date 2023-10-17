requirements：
python 3.6.8
pytorch 1.6.0
transformers 4.18.0
datasets  1.8.0
CUDA version :10.1


命令行输入python T5-finetune.py 即可开始训练
开始训练前请先准备好数据，humor数据集已经在humor_data文件夹中
额外引入的开源数据集dailydialog，运行data目录下的get_dailydialog.py就可以得到数据


data目录下的organize_humor_data,py是用来把测试数据处理成我们需要的格式，用该脚本处理数据后，
命令行输入python predict.py 即可开始预测。

预测结果保存在test_result文件夹下，该目录下的post_process_test_result.py脚本用来把预测结果处理成需要提交的格式。