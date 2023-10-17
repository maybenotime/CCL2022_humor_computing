# coding utf-8
import os
import chardet

# 获得所有txt文件的路径,传入文件所在文件夹路径
def find_all_file(path: str) -> str:
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.ass'):
                fullname = os.path.join(root, f)
                yield fullname
            pass
        pass
    pass

# 判断是不是utf-8编码方式
def judge_coding(path: str) -> dict:
    with open(path, 'rb') as f:
        c = chardet.detect(f.read())

    if c != 'utf-8':  # 改为 c != 'utf-8'
        return c

# 修改文件编码方式
def change_to_utf_file(path: str):
    for i in find_all_file(path):
        c = judge_coding(i)
        if c:
            change(i, c['encoding'])
            print("{} 编码方式已从{}改为 utf-8".format(i, c['encoding']))

def change(path: str, coding: str):
    with open(path, 'r', encoding=coding) as f:
        text = f.read()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# 查看所有文件编码方式
def check(path: str):
    for i in find_all_file(path):
        with open(i, 'rb') as f:
            print(chardet.detect(f.read())['encoding'], ': ', i)

def main():
    my_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/pretrain_data/captions'
    change_to_utf_file(my_path)
    check(my_path)

if __name__ == '__main__':
    main()
