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


#ratio = float(sys.argv[1])
#noise = float(sys.argv[2])
#emotion = sys.argv[1]

nz = 100    #zの次元数
repeat = 6  #画像生成枚数（繰り返し数）

model_root = '../model/'  #感情極性分類モデルのパス
"""
output_csv = 'resource/mean_z%s.csv'
input_csv = 'resource/z%s.csv'
"""
output_images = 'generated_images/degree/%s'
model_file = 'generate_model/all_gen.h5'

#感情極性分類モデルの定義
model = [model_root + 'mean.npy', model_root + 'deploy.prototxt', model_root + 'finetuned.caffemodel', model_root + 'synset_words.txt']
cls = Feature.Classify(model)

#出力csvの定義
#w = csv.writer(open(output_csv%'','w'))
#pos_w = csv.writer(open(output_csv%'_positive','w'))
#neg_w = csv.writer(open(output_csv%'_negative','w'))

#入力csvの定義
#r = csv.reader(open(output_csv%'','w'))
#pos_r = csv.reader(open(input_csv%'_positive','r'))
#neg_r = csv.reader(open(input_csv%'_negative','r'))


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
	
	
def calc_mean(input_r):
    i = 0
    data = np.zeros((1,100),dtype=np.float32)
    for d in input_r:
        d.pop(0)    #画像名前を除去
        i += 1
        data[0] += map(float,d)    #dがｓｔｒのため
        
    data[0] = data[0] / i
    return data


#学習済みモデルの読み込み
gen = Generator()
serializers.load_hdf5(model_file, gen)


"""
pos_z = calc_mean(pos_r)
neg_z = calc_mean(neg_r)
mean_z = pos_z - neg_z
"""


vissize = 1 #一度の生成画像数
pos_count = 0
neg_count = 0

z1 = (np.random.uniform(-1, 1, (vissize, 100)).astype(np.float32)) #生成画像を決定する初期ノイズ
z2 = (np.random.uniform(-1, 1, (vissize, 100)).astype(np.float32)) #生成画像を決定する初期ノイズ
mean = z1 - z2

rnd = (np.random.uniform(-0.5, 0.5, (1, 100)).astype(np.float32)) #平均ノイズに対するランダムノイズ

for i in range(0,11,1):
    i *= 0.1

    out_file = '%f.png'%i

    z = (mean * i) #平均ノイズベクトル
    

    z = z+ z2

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
    print i,score

    #画像をモデルにつっこんで与えられたらべるごとでzのmeanをとる

"""
i = 0   #iの初期化、whileの最後で(+)処理してる
#while(True):
for i in range(repeat):

    out_file = '%d.png'%i

    z = (mean_z * ratio) #平均ノイズベクトル
    rnd = (np.random.uniform(-(1+noise), (1+noise), (1, 100)).astype(np.float32)) #平均ノイズに対するランダムノイズ

    z = z+rnd

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

    #画像をモデルにつっこんで与えられたらべるごとでzのmeanをとる
"""

