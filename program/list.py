import os

dir = 'resource/images'
filename = 'resource/finetune/%s_list.txt'

train = open(filename%'train','w')
test = open(filename%'test','w')

dir = os.path.abspath(dir)
print dir

emotions = os.listdir(dir)

for i in range(len(emotions)):
    d = dir + '/' + emotions[i]
    images = os.listdir(d)
    
    lists = []
    for im in images:
        if 'orig.jpg' in im:
            lists.append(im)
    
    
    tr = lists[:len(lists)/10*8]
    te = lists[len(lists)/10*8:]
    
    for image in tr:
        path = d + '/' + image
        train.write('%s %d\n'%(path,i))
        train.write('%s %d\n'%(path.replace('orig','flip'),i))
        train.write('%s %d\n'%(path.replace('orig','crop1'),i))
        train.write('%s %d\n'%(path.replace('orig','crop2'),i))
        
    for image in te:
        path = d + '/' + image
        test.write('%s %d\n'%(path,i))
        
train.close()
test.close()
    
    
