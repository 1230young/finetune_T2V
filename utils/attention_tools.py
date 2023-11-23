import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt


def print_top_k(t,k=10):
    t=t.reshape(-1)
    temp,_=torch.sort(t,descending=True)
    temp=temp.detach().cpu().numpy()
    
    plt.hist(temp, bins='auto', density=True)
    plt.savefig("output/hist.png")
    # print(temp.shape)
    # print(temp[:k])

def nn_convSobel(im,device):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False, padding=1)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float16')/9
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device)
    # 对图像进行卷积操作
    detect = conv_op(im)
    # 将输出转换为图片格式
    return detect

def attention_map_deflicker(Map,h=576,w=1024,num_frames=16,scale=None,threshold=0.2):
    should_unsqueeze=False
    should_reshape=False
    map=Map.clone()
    if len(map.shape)==5 and map.shape[0]==1:
        map=map.squeeze(0)
    assert len(map.shape)==4, "map should be 4D tensor"
    # reshape to ( _,frame, source, target)
    if map.shape[1]!=num_frames:
        should_reshape=True
        map=map[None, :].reshape((-1, num_frames) + map.shape[1:]).permute(0, 2, 1, 3, 4).reshape(-1, num_frames, map.shape[2], map.shape[3])

    device=map.device
    attn_map=map.clone()
    b,f,token_num,prompt_token_num=attn_map.shape
    b=1
    attn_map=attn_map[:,:,:,1:].sum(-1).sum(0).unsqueeze(0)
    if scale==None:
        scale=round(math.sqrt((h*w) / attn_map.shape[2]))
    h = h // scale
    w = w // scale
    attn_map= attn_map.reshape(b,f,h, w) 
    for i in range(b):
        for j in range(f):
            attn_map[i][j]=attn_map[i][j]/attn_map[i][j].max()
    attn_map_pad=torch.cat([attn_map[:,0,:,:].unsqueeze(1),attn_map,attn_map[:,-1,:,:].unsqueeze(1)],dim=1)
    attn_map_residual=attn_map_pad[:,1:-1,:,:]-0.5*attn_map_pad[:,2:,:,:]-0.5*attn_map_pad[:,:-2,:,:]
    attn_map_residual[:,0,:,:]=attn_map_residual[:,0,:,:]*2
    attn_map_residual[:,-1,:,:]=attn_map_residual[:,-1,:,:]*2
    
    attn_map_residual=nn_convSobel(attn_map_residual.reshape(b*f,1,h,w),device).reshape(b,f,h,w)
    mask=torch.where(attn_map_residual>threshold,torch.ones_like(attn_map_residual),torch.zeros_like(attn_map_residual))
    mask=mask.reshape(b,f,token_num).unsqueeze(3)
    mask=mask.repeat(1,1,1,prompt_token_num)
    map_smooth=map.clone()
    for i in range(b):
        for j in range(f):
            map_smooth[i][j]=map_smooth[i][j]/map_smooth[i][j].max()

    map_smooth=torch.cat([map_smooth[:,1,:,:].unsqueeze(1),0.5*map_smooth[:,0:-2,:,:]+0.5*map_smooth[:,2:,:,:],map_smooth[:,-1,:,:].unsqueeze(1)],dim=1)
    for i in range(b):
        for j in range(f):
            map_smooth[i][j]=map_smooth[i][j]*map[i][j].max()

    attn_map_smooth=torch.where(mask==1,map_smooth,map)
    # print(attn_map_smooth. shape)
    # print(map.shape)
    if should_reshape:
        attn_map_smooth=attn_map_smooth.reshape((-1, Map.shape[1],num_frames) + attn_map_smooth.shape[2:]).permute(0, 2, 1, 3, 4).reshape(Map.shape)

    if should_unsqueeze:
        attn_map_smooth=attn_map_smooth.unsqueeze(0)
    

    return attn_map_smooth






