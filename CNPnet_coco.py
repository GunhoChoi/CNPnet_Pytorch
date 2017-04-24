# Implementation of "Convolutional Neural Pyramid for Image Processing"
# Code by GunhoChoi
# https://discuss.pytorch.org/t/autogradable-image-resize/580

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import collections
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

# Hyperparameters

N = 4
S = 3
lr= 1e-5
epochs = 10
num_gpus = 4
batch_size = 8 

num_channel= 64
input_channel = 3
mapping_channel = 16

# input pipeline

img_dir ="./images/"
img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
											transforms.Scale(size=320),
											transforms.RandomHorizontalFlip(),
											transforms.RandomCrop(size=(240,320)),
											transforms.ToTensor(),
											]))
img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

# Feature Extraction Module

class Feature_Extraction(nn.Module):
	def __init__(self,channel_in=num_channel):
		super(Feature_Extraction, self).__init__()
		self.conv_1 = nn.Conv2d(channel_in,num_channel,kernel_size=3,padding=1)
		self.bn_1 = nn.BatchNorm2d(num_features=num_channel)
		self.prelu_1 = nn.PReLU()
		self.conv_2 = nn.Conv2d(num_channel,num_channel,kernel_size=3,padding=1)
		self.bn_2 = nn.BatchNorm2d(num_features=num_channel)
		self.prelu_2 = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self,img):
		out = self.conv_1(img)
		out = self.bn_1(out)
		out = self.prelu_1(out)
		out = self.conv_2(out)
		out = self.bn_2(out)
		out = self.prelu_2(out)
		down_out = self.maxpool(out)
		return out, down_out


class Mapping(nn.Module):
	def __init__(self):
		super(Mapping, self).__init__()
		self.conv_1 = nn.Conv2d(num_channel,mapping_channel,kernel_size=3,padding=1)
		self.bn_1 = nn.BatchNorm2d(num_features=mapping_channel)
		self.prelu_1 = nn.PReLU()
		self.conv_2 = nn.Conv2d(mapping_channel,mapping_channel,kernel_size=3,padding=1)
		#self.conv_2 = nn.Conv2d(mapping_channel,mapping_channel,kernel_size=3,padding=1)
		#self.conv_2 = nn.Conv2d(mapping_channel,mapping_channel,kernel_size=3,padding=1)
		self.conv_3 = nn.Conv2d(mapping_channel,num_channel,kernel_size=3,padding=1)
		self.bn_3 = nn.BatchNorm2d(num_features=num_channel)
		self.prelu_3 = nn.PReLU()

	def forward(self,img):
		out = self.conv_1(img)
		out = self.bn_1(out)
		out = self.prelu_1(out)
		for i in range(S):
			out = self.conv_2(out)
		out = self.conv_3(out)
		out = self.bn_3(out)
		out = self.prelu_3(out)
		return out


class Reconstruction(nn.Module):
	def __init__(self):
		super(Reconstruction,self).__init__()
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
		self.conv_1 = nn.Conv2d(num_channel,num_channel,kernel_size=3,padding=1)
		self.bn_1 = nn.BatchNorm2d(num_features=num_channel)
		self.prelu_1 = nn.PReLU()

	def forward(self,img,img_under):
		h,w = img_under.size()[2:]
		out_under = self.upsample(img_under)
		out_under = self.conv_1(out_under)		
		out_under = self.bn_1(out_under)
		out_under = self.prelu_1(out_under)
		out = img + out_under
		return out


class Adjustment(nn.Module):
	def __init__(self):
		super(Adjustment,self).__init__()
		self.prelu_1 = nn.PReLU()
		self.conv_1 = nn.Conv2d(num_channel,num_channel,kernel_size=3,padding=1)
		self.bn_1 = nn.BatchNorm2d(num_features=num_channel)
		self.prelu_2 = nn.PReLU()
		self.conv_2 = nn.Conv2d(num_channel,input_channel,kernel_size=3,padding=1)

	def forward(self,img):
		out = self.prelu_1(img)
		out = self.conv_1(out)
		out = self.bn_1(out)
		out = self.prelu_2(out)
		out = self.conv_2(out)
		return out


def list_to_dict(instance_list,module=False):
	dict_instance = collections.OrderedDict()
	for i in instance_list:
		if module:	
			dict_instance[i] = nn.DataParallel(module,device_ids=[i for i in range(num_gpus)]).cuda()
		else:
			dict_instance[i] = Variable(None).cuda()
	return dict_instance

# image postprocess

def image_postprocess(tensor):
	img = tensor.clone()
	img[img>1] = 1    
	img[img<0] = 0
	img = 2*(img-0.5)
	return img

def image_down(var):
	img = var.clone()
	h,w = img.size()[2:]
	transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Scale(h//2),
				transforms.ToTensor()
				])
	out = transform(img[0])
	for i in range(batch_size-1):
		out_0 = transform(img[i+1])
		out = torch.cat([out,out_0],0)
	out = out.view(batch_size,input_channel,h//2,w//2)
	return out

# module to dict

f_list = ["f"+str(i) for i in range(N-1)]
m_list = ["m"+str(i) for i in range(N-1)]
r_list = ["r"+str(i) for i in range(N-1)]
a_list = ["a"+str(i) for i in range(N-1)]

f_dict = list_to_dict(f_list,Feature_Extraction())
m_dict = list_to_dict(m_list,Mapping())
r_dict = list_to_dict(r_list,Reconstruction())
a_dict = list_to_dict(a_list,Adjustment())
f_dict["f0"] = Feature_Extraction(input_channel).cuda()

# output to dict

f_out = ["f"+str(i)+"_"+str(j) for i in range(N-1) for j in range(2)]
m_out = ["m"+str(i) for i in range(N-1)]
r_out = ["r"+str(i) for i in range(N-1)]
a_out = ["a"+str(i) for i in range(N-1)]

f_out_dict = list_to_dict(f_out)
m_out_dict = list_to_dict(m_out)
r_out_dict = list_to_dict(r_out)
a_out_dict = list_to_dict(a_out)

# gather all parameters for training

params = []
for i in f_dict:
	params+=list(f_dict[i].parameters())
for i in m_dict:
	params+=list(m_dict[i].parameters())
for i in r_dict:
	params+=list(r_dict[i].parameters())
for i in a_dict:
	params+=list(a_dict[i].parameters())

# loss function & optimizer

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(params,lr=lr)

# try restoring model 

try:
	restore = torch.load('./model/denoise_cnp_64.pkl')
	for i in range(N):
		f_dict[f_list[i]] = restore[i]
		m_dict[m_list[i]] = restore[N+i]
		r_dict[r_list[i]] = restore[2*N+i]
		a_dict[a_list[i]] = restore[3*N+i]
	print("\n--------model restored--------\n")
except:
	print("\n--------model not restored--------\n")
	pass

# training

for k in range(epochs):
	for _,(image,label) in enumerate(img_batch):
		h,w = image.size()[2:]

		# add noise to image
		
		clean_image = image
		noise = np.random.normal(0.5,0.1,[batch_size,input_channel,h,w])
		noise[noise<0.4]=0
		noise[noise>=0.4]=1
		noise = torch.from_numpy(noise).type_as(torch.FloatTensor())
		image = torch.mul(image, noise)

		# set variables required

		image = Variable(image).cuda()
		img_0 = Variable(clean_image).cuda()
		#img_1 = Variable(image_down(img_0.cpu().data)).cuda()
		#img_2 = Variable(image_down(img_1.cpu().data)).cuda()
		#img_3 = Variable(image_down(img_2.cpu().data)).cuda()
		#img_4 = Variable(image_down(img_3.cpu().data)).cuda()

		zero_tensor = Variable(torch.zeros(batch_size,num_channel,h//2**N,w//2**N)).cuda()
		optimizer.zero_grad()

		# initial values

		f_out_dict["f0_0"],f_out_dict["f0_1"] = f_dict["f0"](image)
		m_out_dict["m0"] = m_dict[m_list[0]](f_out_dict["f0_0"])

		# Feature & Mapping

		for i in range(N-1):
			j = i+1
			f_out_dict[f_out[2*j]],f_out_dict[f_out[2*j+1]] = f_dict[f_list[j]](f_out_dict[f_out[2*i+1]])
			m_out_dict[m_out[j]] = m_dict[m_list[j]](f_out_dict[f_out[2*j]])
		
		# Reconstruction

		for i in range(N):
			j=N-i-1 # N-1~0
			if j==N-1:
				r_out_dict[j] = r_dict[r_list[j]](m_out_dict[m_out[j]],zero_tensor)
				a_out_dict[j] = a_dict[a_list[j]](r_out_dict[j])
			else:
				r_out_dict[j] = r_dict[r_list[j]](m_out_dict[m_out[j]],r_out_dict[j+1])
				a_out_dict[j] = a_dict[a_list[j]](r_out_dict[j])

		loss_0 = loss_func(a_out_dict[0],img_0)
		#loss_1 = loss_func(a_out_dict[1],img_1)
		#loss_2 = loss_func(a_out_dict[2],img_2)
		#loss_3 = loss_func(a_out_dict[3],img_3)
		#loss_4 = loss_func(a_out_dict[4],img_4)

		total_loss = loss_0 #+ loss_1 + loss_2 + loss_3 #+ loss_4
		total_loss.backward()
		optimizer.step()

		# model save

		save_list = [f_dict[f_list[i]] for i in range(N)] + [m_dict[m_list[i]] for i in range(N)]\
				  + [r_dict[r_list[i]] for i in range(N)] + [a_dict[a_list[i]] for i in range(N)] 
		
		if k % 1==0 and _%400==0:
			v_utils.save_image(image_postprocess(clean_image),"./output/clean_image/clean_image_{}_{}.png".format(k,_),nrow=4)
			v_utils.save_image(image_postprocess(image.cpu().data),"./output/noise_image/noise_image_{}_{}.png".format(k,_),nrow=4)
			v_utils.save_image(image_postprocess(a_out_dict[0].cpu().data),"./output/result_image/output_{}_{}.png".format(k,_),nrow=4)
			torch.save(save_list,"./model/denoise_cnp_64.pkl")
			print("{}th iteration {}th batch loss: {}".format(k,_,total_loss.data))
