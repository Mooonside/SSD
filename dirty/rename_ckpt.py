import tensorflow as tf
import numpy as np

from tf_utils import rename_vars_in_ckpt


old_names = [i.rstrip('\n') for i in open('org.txt').readlines()]
new_names = [i.rstrip('\n') for i in open('new.txt').readlines()]

name_map = dict([(old_names[idx], new_names[idx]) for idx in range(len(old_names))])

rename_vars_in_ckpt(ckpt_path='/home/yifeng/SSD/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt',
                    name_map=name_map,
                    output_path='./ckpts/SSD_VGG300_120000.ckpt')