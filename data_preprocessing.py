from __future__ import print_function
import torch
from torchvision import datasets, transforms

from data_init import Data_Initialization


def load_training_data(filePath , batch_size , upscale_factor):
	crop_size = 256 - (256 % upscale_factor)
	train_data = Data_Initialization(filePath , input_transform(crop_size , upscale_factor) , target_transform(crop_size , upscale_factor))
	return torch.utils.data.DataLoader(dataset=train_data , batch_size=batch_size , shuffle=True)

def load_testing_data(filePath , batch_size , upscale_factor):
	crop_size = 256 - (256 % upscale_factor)
	test_data = Data_Initialization(filePath , input_transform(crop_size , upscale_factor) , target_transform(crop_size , upscale_factor))
	return torch.utils.data.DataLoader(dataset = test_data , batch_size=batch_size , shuffle=False)

def input_transform(crop_size , upscale_factor):
	return transforms.Compose([transforms.CenterCrop(crop_size) , transforms.Resize(int(crop_size/upscale_factor)) , transforms.ToTensor(),])
	#return transforms.Compose([transforms.CenterCrop(crop_size) , transforms.Resize(int(crop_size)) , transforms.ToTensor(),])
def target_transform(crop_size , upscale_factor):
	return transforms.Compose([transforms.CenterCrop(crop_size) , transforms.Resize(int(crop_size)) , transforms.ToTensor(),])