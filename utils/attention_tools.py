import torch
from torch import nn
import numpy as np

def nn_convSobel(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/9
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    detect = conv_op(im)
    # 将输出转换为图片格式
    return detect

def attention_map_deflicker(map,scale,h,w,device):
    attn_map=map.clone()
    b,f,token_num,prompt_token_num=attn_map.shape
    attn_map=attn_map[:,:,:,1:].sum(-1).sum(0).unsqueeze(0)
    h = h // scale
    w = w // scale
    attn_map= attn_map.reshape(b,f,h, w) 
    for i in range(b):
        for j in range(f):
            attn_map[i][j]=attn_map[i][j]/attn_map[i][j].max()
    attn_map_pad=torch.cat([attn_map[:,0,:,:].unsqueeze(1),attn_map,attn_map[:,-1,:,:].unsqueeze(1)],dim=1)
    attn_map_residual=attn_map[:,1:-1,:,:]-0.5*attn_map[:,2:,:,:]-0.5*attn_map[:,:-2,:,:]
    attn_map_residual[:,0,:,:]=attn_map_residual[:,0,:,:]*2
    attn_map_residual[:,-1,:,:]=attn_map_residual[:,-1,:,:]*2
    attn_map_residual=nn_convSobel(attn_map_residual.reshape(b*f,1,h,w)).reshape(b,f,h,w)
    threshold=0.5
    mask=torch.where(attn_map_residual>threshold,torch.ones_like(attn_map_residual),torch.zeros_like(attn_map_residual))
    mask=mask.reshape(b,f,token_num)
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






