import sys
sys.path.append('/raid/pengyuyang/code/Text-To-Video-Finetuning')
import clip
import torch
from einops import rearrange
from torchvision import transforms
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
def CLIPSIM(text, video,device='cuda:0'):
    #b,f,c,h,w
    model, preprocess = clip.load('ViT-B/32', device)
    if len(video.shape) == 5:
        assert len(text)==video.shape[0], "text and video must have same batch size"
        similarity=0.0
        for n,vid in enumerate(video):
            text_one = clip.tokenize(text[n]).to(device)
            vid = vid.to(device)
            sim=0.0
            for i in range(vid.shape[0]):
                
                with torch.no_grad():
                    transform=transforms.ToPILImage()
                    image=preprocess(transform(vid[i])).unsqueeze(0).to(device)
                    text_features = model.encode_text(text_one)
                    image_features = model.encode_image(image)
                    sim+=torch.nn.functional.cosine_similarity(image_features, text_features).item()
            sim/=vid.shape[0]
            similarity+=sim
        similarity/=video.shape[0]
        return similarity            
    elif len(video.shape) == 4:
        if isinstance(text, list):
            assert len(text)==1, "text and video must have same batch size"
            text=text[0]
        video= rearrange(video, 'c f h w ->f c h w')
        text = clip.tokenize(text).to(device)
        vid = vid.to(device)
        sim=0.0
        for i in range(video.shape[0]):
            
            with torch.no_grad():
                transform=transforms.ToPILImage()
                image=preprocess(transform(video[i])).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                sim+=torch.nn.functional.cosine_similarity(image_features, text_features).item()
        sim/=vid.shape[0]
        return sim

    else:
        raise ValueError("video must be 4-dim or 5-dim tensor")

if __name__ == '__main__':
    Dataset=VideoJsonDataset(width=1024,height=576,use_bucketing=True,sample_start_idx=1,fps=24,frame_step=1,n_sample_frames=16,json_path="/raid/pengyuyang/code/Text-To-Video-Finetuning/datasets/potat1/potat1_blip2.json",normalize=False)
    dataloader=torch.utils.data.DataLoader(Dataset,batch_size=8,shuffle=True,num_workers=0)
    for batch in dataloader:
        video=batch['pixel_values']
        text=batch["text_prompt"]
        print(CLIPSIM(text,video))
        break