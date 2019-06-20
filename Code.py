# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:47:26 2019

@author: Shrishti D Hore
"""

import numpy as numpy 
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os 
import zipfile
 
def main():
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exits(local_zip_file):
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    model_fn = 'tensorflow_inception_graph.pb'
    
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')
    imagnet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagnet_mean, 0)
    tf.import_graph_def(graph_def, {'input : t_preprocessed'})
    
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+ ':0').get_shape()[-1]) for name in layers]
    
    def render_deepdream(t_obj, img0 = img_noise, iter_n = 10, step= 1.5, octave_n = 4, octave_scale= 1.4 ):
        t_score = tf.reduce_mean(t_score, t_input)[0]
        
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(low, hw)
            img = lo
            octaves.append(hi)
            
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img +=g*(step/(np.abs(g).mean()+1e-7))
            
        showarray(img/255.0)
        
    print('Numbers of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
    
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    
    img0 = PIL.Image.open('pilatus800.jpg')
    img0 = np.float32(img0)
    
    render_deepdream(tf.square(T('mixed4c')), img0)
    
if __name__ == '__main__':
    main()
