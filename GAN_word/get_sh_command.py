import os
import torch.utils.data as D
import random
import string
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

'''
    todo：产生 .sh
'''
"""
example:
python tt.test_single_writer.allWids.py --model_id 13600 --wid 049 --gpu_id 2
根据每一个wid对应的groundtruth以及words，生成图像。
"""
#
# model_id = 13600
# gpu_id = 2
#
#
# """ 000 -> 671  中间有空的"""
# wid_list = list()
# for i in range(671 + 1):
#     if len(str(i)) == 1:
#         wid_list.append("00"+str(i))
#     elif len(str(i)) == 2:
#         wid_list.append("0"+str(i))
#     else:
#         wid_list.append(str(i))
#
# file_obj = open('./sh_command/gen_eachWid_img.sh','w')
# for i in wid_list:
#     command = "python tt.test_single_writer.allWids.py --model_id {} --wid {} --gpu_id {}".format(model_id, i, gpu_id)
#     file_obj.write(command+'\n')
# file_obj.close()



'''
    todo:根据每一个作者，计算生成img和数据集中img的fid
'''
"""
example:
python fid_score_crop64x64.py --gpu 2 --batch-size=256 /home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/fid_cal/epoch_13600_eachWidRes/000 /home/WeiHongxi/WangYiming/data/iamDatabase/words_wid/000

"""

# gpu_id = 2
#
# output_save_txt_path = '/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/fid_folder/fid_res/fid_score_output_Ganwriting_iam_13600.txt'
#
# train_set_path = '/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/Groundtruth/gan.iam.tr_va.gt.filter27'
# test_set_path = '/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/Groundtruth/gan.iam.test.gt.filter27'
#
# with open(train_set_path, 'r') as _f:
#     train_set = _f.readlines()
#
# with open(test_set_path, 'r') as _f:
#     test_set = _f.readlines()
#
# full_set = train_set + test_set
# full_list = list()
#
# def filter_info_img(line):
#     wid = line.split(',')[0]
#     img_info = line.split(',')[-1]
#     name = img_info.split(' ')[0]
#     word = img_info.split(' ')[1]
#     word = word.replace("\n", '')
#     return wid, name, word
# def dedupe(items):
#     seen = set()
#     for item in items:
#         if item not in seen:
#             yield item
#             seen.add(item)
#
# for i in full_set:
#      iwid, iname, iword = filter_info_img(i)
#      full_list.append(iwid)
# full_list = list(dedupe(full_list))
#
#
# file_obj = open('./sh_command/cal_each_wid.sh','w')
# for i in full_list:
#     command = "python fid_score_crop64x64.py " \
#               "--gpu {} " \
#               "--batch-size=256 " \
#               "/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/fid_cal/epoch_13600_eachWidRes/{} " \
#               "/home/WeiHongxi/WangYiming/data/iamDatabase/words_wid/{}".format(gpu_id, i, i) + ' >> ' +output_save_txt_path
#
#     file_obj.write(command+'\n')
# file_obj.close()

'''
    todo : 为每一个wid（seen unseen）生成 inv oov图像的 .sh文件。
'''
'''
    example:
    python tt.test_each_wid_generate_iv_oov.py --model_id 16200 --wid 049 --gpu_id 2 --wordtype inv --styletype seen
'''
# gpu_id = 2
# model_id = 16200
#
# train_set_path = '/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/Groundtruth/gan.iam.tr_va.gt.filter27'
# test_set_path = '/home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/Groundtruth/gan.iam.test.gt.filter27'
#
# with open(train_set_path, 'r') as _f:
#     train_set = _f.readlines()
#
# with open(test_set_path, 'r') as _f:
#     test_set = _f.readlines()
#
# def filter_info_img(line):
#     wid = line.split(',')[0]
#     img_info = line.split(',')[-1]
#     name = img_info.split(' ')[0]
#     word = img_info.split(' ')[1]
#     word = word.replace("\n", '')
#     return wid, name, word
#
# def dedupe(items):
#     seen = set()
#     for item in items:
#         if item not in seen:
#             yield item
#             seen.add(item)
#
# train_list = list()
# test_list = list()
#
# for i in train_set:
#      iwid, iname, iword = filter_info_img(i)
#      train_list.append(iwid)
#
# train_list = list(dedupe(train_list))
# for i in test_set:
#      iwid, iname, iword = filter_info_img(i)
#      test_list.append(iwid)
# test_list = list(dedupe(test_list))
#
# file_obj = open('./sh_command/gen_eachWid_inv_oov_img.sh','w')
# for i in train_list:
#     command1 = "python tt.test_each_wid_generate_iv_oov.py " \
#               "--model_id {} " \
#               "--wid {} " \
#               "--gpu_id {} " \
#               "--wordtype inv " \
#               "--styletype seen".format(model_id, i, gpu_id)
#     command2 = "python tt.test_each_wid_generate_iv_oov.py " \
#               "--model_id {} " \
#               "--wid {} " \
#               "--gpu_id {} " \
#               "--wordtype oov " \
#               "--styletype seen".format(model_id, i, gpu_id)
#     file_obj.write(command1 + '\n')
#     file_obj.write(command2 + '\n')
#
# for i in test_list:
#     command1 = "python tt.test_each_wid_generate_iv_oov.py " \
#               "--model_id {} " \
#               "--wid {} " \
#               "--gpu_id {} " \
#               "--wordtype inv " \
#               "--styletype unseen".format(model_id, i, gpu_id)
#     command2 = "python tt.test_each_wid_generate_iv_oov.py " \
#               "--model_id {} " \
#               "--wid {} " \
#               "--gpu_id {} " \
#               "--wordtype oov " \
#               "--styletype unseen".format(model_id, i, gpu_id)
#     file_obj.write(command1 + '\n')
#     file_obj.write(command2 + '\n')
# file_obj.close()

'''
    todo : 为每一个wid（seen unseen）计算其 inv oov图像与真实图像的的 .sh文件。
'''
'''
    example:
    python fid_score_crop64x64.py --gpu 2 --batch-size=256 
    /home/WeiHongxi/WangYiming/Project_GANwriting/research-GANwriting-master/fid_cal/epoch_16200_each_wid_inv_oov/seen_000_inv 
    /home/WeiHongxi/WangYiming/data/iamDatabase/words_wid/000
'''
gpu_id = 1



train_set_path = 'Groundtruth/gan.iam.tr_va.gt.filter27'
test_set_path = 'Groundtruth/gan.iam.test.gt.filter27'

with open(train_set_path, 'r') as _f:
    train_set = _f.readlines()

with open(test_set_path, 'r') as _f:
    test_set = _f.readlines()

def filter_info_img(line):
    wid = line.split(',')[0]
    img_info = line.split(',')[-1]
    name = img_info.split(' ')[0]
    word = img_info.split(' ')[1]
    word = word.replace("\n", '')
    return wid, name, word

def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

train_list = list()
test_list = list()

for i in train_set:
     iwid, iname, iword = filter_info_img(i)
     train_list.append(iwid)

train_list = list(dedupe(train_list))
train_list.sort()
for i in test_set:
     iwid, iname, iword = filter_info_img(i)
     test_list.append(iwid)
test_list = list(dedupe(test_list))
test_list.sort()
file_obj = open('fid_folder/cal_fid_eachWid_inv_oov.sh','w')
for i in train_list:
    command1 = "python fid_score_crop64x64.py --gpu {} --batch-size=256 --in_oov={} " \
               "/home/WeiHongxi/WangHeng/project/research-GANwriting-master/test_single_writer.4_scenarios_average/8400/res_1.in_vocab_tr_writer" \
               "/{} " \
               "/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words_wid/{}".format(gpu_id,1, i, i)
    command2 = "python fid_score_crop64x64.py --gpu {} --batch-size=256 --in_oov={} " \
               "/home/WeiHongxi/WangHeng/project/research-GANwriting-master/test_single_writer.4_scenarios_average/8400/res_3.oo_vocab_tr_writer" \
               "/{} " \
               "/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words_wid/{}".format(gpu_id, 2,i, i)
    file_obj.write(command1 + '\n')
    file_obj.write(command2 + '\n')

for i in test_list:
    command1 = "python fid_score_crop64x64.py --gpu {} --batch-size=256 --in_oov={} " \
               "/home/WeiHongxi/WangHeng/project/research-GANwriting-master/test_single_writer.4_scenarios_average/8400/res_2.in_vocab_te_writer" \
               "/{} " \
               "/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words_wid/{}".format(gpu_id, 1, i, i)
    command2 = "python fid_score_crop64x64.py --gpu {} --batch-size=256 --in_oov={} " \
               "/home/WeiHongxi/WangHeng/project/research-GANwriting-master/test_single_writer.4_scenarios_average/8400/res_4.oo_vocab_te_writer" \
               "/{} " \
               "/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words_wid/{}".format(gpu_id,2, i, i)
    file_obj.write(command1 + '\n')
    file_obj.write(command2 + '\n')

file_obj.close()
