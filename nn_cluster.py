import torch
from torch.autograd import Variable
from torch import nn
import itertools
import csv
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import cv2
from scipy.cluster.hierarchy import fcluster
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import pandas as pd
import math
import os

file = r'/fs/janus-scratch/carlos/cs4-pre-deliverable/umdfaces_batch1_ultraface.csv'
df = pd.read_csv(file)
file_feat = r'/fs/janus-scratch/carlos/cs4-pre-deliverable/umdfaces_batch1_rajeev_spring17_deep_features.csv'
df_feat = pd.read_csv(file_feat)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
num_epochs=0
loss_mini_batch=0
loss_val= []
iterations=[]

def convert2tensor(x):
    x = torch.cuda.FloatTensor(x)
    return x
def feat_extract(record):
	
	
	field = 'DEEPFEATURE_'
	img_feat = []
	i=1
	while i<=512 :
		img_feat.append(float(record[field+str(i)]))
		i+=1
	return img_feat

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_in,H_out, I_in,I_out, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Sequential(
		nn.Linear(D_in, H_in),
		nn.SELU(),
		nn.Linear(H_in, H_out),
		nn.SELU(),
		nn.Linear(H_out, I_in),
		nn.SELU(),
		nn.Linear(I_in, I_out),
		nn.SELU(),
		nn.Linear(I_out,D_out))
        
        
        self.linear2 = nn.Sequential(
		nn.Linear(D_in, H_in),
		nn.SELU(),
		nn.Linear(H_in, H_out),
		nn.SELU(),
		nn.Linear(H_out, I_in),
		nn.SELU(),
		nn.Linear(I_in, I_out),
		nn.SELU(),
		nn.Linear(I_out,D_out))
	

    def forward(self, x1, x2):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        r1 = self.linear1(x1)
        r2 = self.linear2(x2)
        return r1*r2


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H_in,H_out, I_in,I_out, D_out =1024 ,512, 256,128,64,32, 1


# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H_in,H_out, I_in,I_out, D_out)
model.cuda()
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
optimizer.zero_grad()
def iterate_nn(x1,x2,y):
	'''
	a batch of 32 pairs is passed through the model
	the avg loss for the batch is calculated and the error is 
	BACK PROPAGATED ONCE
	'''
	#print "iterating"
		
	x1 = convert2tensor(x1)
	x2 = convert2tensor(x2)
	y = convert2tensor(y)
	x1 = Variable(x1)
	x2 = Variable(x2)
	y = Variable(y)
	x1 = torch.unsqueeze(x1,0)
	x2 = torch.unsqueeze(x2,0)
	#y = torch.unsqueeze(y,0)
	#print "sizes of x1 x2 y"
	#print x1.size()
	#print x2.size()
	#print y.size()
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	sim = cos(x1,x2)
	sim = torch.unsqueeze(sim, 1)
	#print "sim: ",sim
	#print "y::",y
	y_pred = model(x1,x2)
   	y_pred_new = y_pred*sim
	#print "y pred:", y_pred
	#print "y pred new", y_pred_new
	
	loss = criterion(y_pred_new, y)/1024.0 #averaging
	#print "loss: ", loss	    
	loss.backward() #Accumulating gradients
	#ggg=raw_input()
  	return loss.data[0]
	



def get_pair(source):
	result=[]
	all_feat=[]
	global loss_mini_batch
        for p1 in range(len(source)):
                for p2 in range(p1+1,len(source)):
                        
			record1 = (df_feat[df_feat['FILE']== source[p1]])
			record2 = (df_feat[df_feat['FILE']== source[p2]])
			if record1.empty or record2.empty :
				continue
			if source[p1].split('/')[0] == source[p2].split('/')[0] :
				label = [1.0] ## similarity:1
			else: 
				label = [0.0] ## similarity:0
			result.append([source[p1],source[p2],label])
			feature1 = feat_extract(record1)
			feature2 = feat_extract(record2)
			#print label
			loss_mini_batch += iterate_nn(feature1,feature2,label)#we calculate average loss 
			pre_refresh_loss=loss_mini_batch
			if len(result)==1024: 
				optimizer.step()#after a mini-batch is parsed we update weights
				optimizer.zero_grad()#flushing out accumulated gradient
				result=[]
				print "loss after a minibatch is parsed: ", loss_mini_batch
				
				loss_mini_batch=0

	return pre_refresh_loss			


scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

while(num_epochs<20):
 scheduler.step()
 req_sub_id = 0
 
 ids = sorted(os.listdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/'))
 print len(ids)
 all_names = []
 all_labels = []

 while req_sub_id <50  :
    if os.path.isdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/' + ids[req_sub_id]):
	index=0
	imgs = sorted(os.listdir('/fs/janus-scratch/ankan/umdfaces/umdfaces_batch1/' + ids[req_sub_id]))	
	for i in imgs:
		all_names.append(ids[req_sub_id]+'/'+i)	
    req_sub_id+=1

 loss=get_pair(all_names)
 print "appending: ",loss
 print "type:", type(loss)
 print loss_val
 loss_val.append(loss)
 print loss_val
 fff=raw_input()
 iterations.append(num_epochs)
 np.save('loss_'+str(num_epochs)+'.npy', loss_val)
 np.save('epoch_'+str(num_epochs)+'.npy', iterations)
 torch.save(model,'strectchnet_'+str(num_epochs)+'.pt')
 num_epochs+=1
 print "epoch completed: ", num_epochs
 print "starting new epoch"
