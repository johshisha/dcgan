import pickle,sys
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab,argparse


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
import numpy

sys.path.append('/home/dl-box/study/.package/python_util/')
import util


ap = argparse.ArgumentParser()
ap.add_argument("-emo", "--emotion", required=True, help="Emotion")
ap.add_argument("-param", "--parameter", default=0.5, help="Parameter")
ap.add_argument("-g", "--gpu", default=0, help="gpu id")
args = vars(ap.parse_args())

emotion = args['emotion']

out_image_dir = './resource/each_emo/%s_out_images'%emotion
out_model_dir = './resource/each_emo/%s_out_models'%emotion

nz = 100          # # of dim for Z
batchsize=1000
n_epoch=10000
n_train=200000
image_save_interval = 50000

BETA1 = float(args['parameter'])
print 'prams:',BETA1

out_image_dir+='_%s'%BETA1
out_model_dir+='_%s'%BETA1

# read all images

dataset = []
path = 'resource/images/%s'%emotion
images = util.comb_path(path,os.listdir(path))
for fn in images:
    img_bin = np.asarray(Image.open(fn).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
    dataset.append(img_bin)

print len(dataset)

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



class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(4*4*512, 2, wscale=0.02*math.sqrt(4*4*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l




def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))


def train_dcgan_labeled(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=BETA1)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=BETA1)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0,n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = np.zeros((batchsize, 3, 64, 64), dtype=np.float32)
            for j in range(batchsize):
                try:
                    rnd = np.random.randint(len(dataset))
                    rnd2 = np.random.randint(2)
                    img = dataset[rnd]
                    if rnd2==0:
                        x2[j,:,:,:] = (img[:,:,::-1]-128.0)/128.0
                    else:
                        x2[j,:,:,:] = (img[:,:,:]-128.0)/128.0
                except:
                    print 'read image error occured', fs[rnd]
            #print "load image done"
            
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))
            
            # train discriminator
                    
            x2 = Variable(cuda.to_gpu(x2))
            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))
            
            #print "forward done"

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()
            
            #print "backward done"

            if i%image_save_interval==0:
                pylab.rcParams['figure.figsize'] = (16.0,16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
                    pylab.subplot(10,10,i_+1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%d.png'%(out_image_dir, epoch,i))
                
        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train


gpu = int(args['gpu'])
gen = Generator()
dis = Discriminator()

if gpu >= 0:
    xp = cuda.cupy
    cuda.get_device(gpu).use() 

    gen.to_gpu()
    dis.to_gpu()
else:
    xp = np
    

util.make_folder(out_image_dir)
util.make_folder(out_model_dir)

train_dcgan_labeled(gen, dis)
