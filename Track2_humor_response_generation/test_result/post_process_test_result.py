import csv


result_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/test_result/Tal2022_任务2_largev2.csv'


csv_data = []
header = []
with open(result_path) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    header = next(csv_reader)        # 读取第一行每一列的标题 
    for row in csv_reader:
        sen = row[2]
        start_index = sen.find(':')
        row[2] = sen[start_index+1:]
        row[1] = sen[:start_index]
        csv_data.append(row)

with open(result_path,"w",encoding='utf-8',newline='') as w:
    writer = csv.writer(w)
    writer.writerow(header)
    writer.writerows(csv_data)
