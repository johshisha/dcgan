#coding:utf-8

import numpy as np
from PIL import Image
from StringIO import StringIO
import cv2 as cv
import csv, sys, pylab, math, os, pickle

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function
import chainer.functions as F
import chainer.links as L

sys.path.append('/home/dl-box/study/.package/python_util/')
import util,Feature

nz = 100    #zの次元数

emo = 'all'

repeat = 50  #画像生成枚数（繰り返し数）

model_root = '../model/'  #感情極性分類モデルのパス
util.make_folder('generated_images/%s/'%emo)
output_images = 'generated_images/%s/'%emo + '%s'
model_file = 'generate_model/%s_gen.h5'%emo

#感情極性分類モデルの定義
model = [model_root + 'mean.npy', model_root + 'deploy.prototxt', model_root + 'finetuned.caffemodel', model_root + 'synset_words.txt']
cls = Feature.Classify(model)


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 4*4*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization(4*4*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 4, 4))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x



def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

#学習済みモデルの読み込み
gen = Generator()
serializers.load_hdf5(model_file, gen)


vissize = 1 #一度の生成画像数

#i = 0   #iの初期化、whileの最後で(+)処理してる
for i in range(repeat):
#while(True):
    out_file = '%s_%d.png'%(emo,i)

    z = (np.random.uniform(-1, 1, (vissize, 100)).astype(np.float32)) #生成画像を決定する初期ノイズ

    v = list(z[0][:])   
    v.insert(0,out_file)
    #w.writerow(v)

    z = Variable(z)
    x = gen(z, test=True)
    x = x.data

    imdata = x[0].transpose(1,2,0)
    imdata = cv.resize(imdata,(256,256))
    
    #cvで保存用に変換、計算にはimdataを用いる
    cvdata = cv.cvtColor(imdata,cv.COLOR_RGB2BGR)
    cvdata = ((np.vectorize(clip_img)(cvdata[:,:,:])+1)/2)
    cv.imwrite(output_images%out_file,cvdata*255.0)
    
    
    #画像の感情分類
    res = cls.array_classify(imdata)
    scores = cls.category_score(imdata,'array')
    score = scores[0][0]
    print score
        
    i += 1
    
    #画像をモデルにつっこんで与えられたらべるごとでzのmeanをとる


"""
for i_ in range(vissize):
    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
    pylab.subplot(10,10,i_+1)
    pylab.imshow(tmp)
    pylab.axis('off')
pylab.savefig(out_file)
"""


