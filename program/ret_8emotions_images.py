#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, numpy, caffe, time, csv
from random import random, shuffle
from numpy.random import *
from multiprocessing import Pool
import cv2 as cv
sys.path.append('/home/dl-box/study/.package/python_util/')
import util


def read_lists(filename,emotions):
    f = open(filename,'r').read()
    emos = f.split('\n\n')
    anp_dict = {}
    for i in emos:
        emo = i.split('\n')
        for la in xrange(len(emotions)):
            if emo[0] in emotions[la]:
                label = la
                break
        try:
            for anp in emo[1].split(',')[0:-1]:
                anp_dict[anp] = label
        except:
            continue

    return anp_dict

emotions = [None]*8
emotions[0]=['exstacy','joy','serenity']
emotions[1] = ['admiration','trust','acceptance']
emotions[2] = ['terror','fear','apprehension']
emotions[3] = ['amazament','surprise','distraction']
emotions[4] = ['grief','sadness','presiveness']
emotions[5] = ['loathing','disgust','boredom']
emotions[6] = ['rage','anger','annoyance']
emotions[7] = ['vigilance','anticipation','interest']


anp_dict = read_lists('resource/24emotions.txt',emotions)

print len(anp_dict)

dirname = '/media/dl-box/HD-LCU3/study/dataset/sentibank/images'

outdir = 'resource/images/%s'

abs_path = os.path.abspath(dirname)


lists = []
for i in xrange(len(emotions)):
    lists.append([])

for anp,label in anp_dict.items():
    adject = anp.split(' ')[0]
    anp_path = abs_path + '/' + adject + '/' + anp.replace(' ','_')
    try:
        images = util.comb_path(anp_path,os.listdir(anp_path))
        for image in images:
            lists[label].append(image)
    except:
        continue
        
        
        
def each_emotion(i):
    li = lists[i]
    out = outdir%emotions[i][1]
    util.make_folder(out)
    
    j = 0
    while len(os.listdir(out)) < 100000:
        j += 1
        try:
            filename = li[j].split('/')[-1]
            filename = filename.split('.')[0]
            
            if os.path.isfile('%s/%s_orig.jpg'%(out,filename)) and os.path.isfile('%s/%s_flip.jpg'%(out,filename)) and os.path.isfile('%s/%s_crop1.jpg'%(out,filename)) and os.path.isfile('%s/%s_crop2.jpg'%(out,filename)):
                continue
            
            im = cv.imread(li[j])
            flip = cv.flip(im,1)
            
            h = im.shape[0]
            w = im.shape[1]
            size = min(h,w)
            
            cv.imwrite('%s/%s_orig.jpg'%(out,filename), cv.resize(im,(64,64)))
            cv.imwrite('%s/%s_flip.jpg'%(out,filename), cv.resize(flip,(64,64)))
        
            cv.imwrite('%s/%s_crop1.jpg'%(out,filename), cv.resize(im[0:size,0:size],(64,64)))
            cv.imwrite('%s/%s_crop2.jpg'%(out,filename), cv.resize(im[h-size:h,w-size:w],(64,64)))
            
        except:
            print '%s:%d (%s)'%(emotions[i],j, filename)
            pass


p = Pool(8)

p.map(each_emotion, range(8))

    
    
    

