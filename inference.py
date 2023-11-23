# MIT License

# Copyright (c) 2023 Hans Brouwer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import os
import platform
import re
import warnings
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np

# np.set_printoptions(threshold=np.inf)
import torch
from compel import Compel
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline, UNet3DConditionModel, StableDiffusionAdapterPipeline
from einops import rearrange
from torch import Tensor, einsum
from torch.nn.functional import interpolate
from tqdm import trange
import math


from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark
from utils.lora import inject_inferable_lora
from nltk import word_tokenize

import imageio
from diffusers.models.attention_processor import Attention
from utils.chat import askChatGPT
from utils.attention_tools import attention_map_deflicker
def update_layer_names(model,hidden_layer_select = None):
    hidden_layers = {}
    for n, m in model.named_modules():
        if(isinstance(m, Attention)):
            hidden_layers[n] = m
    hidden_layer_names = list(filter(lambda s : "attn2" in s, hidden_layers.keys())) 
    if hidden_layer_select != None:
        hidden_layer_select.update(value="model.diffusion_model.middle_block.1.transformer_blocks.0.attn2", choice=hidden_layer_names)
    return hidden_layers,hidden_layer_names

def get_attn(emb, ret, f=16):
    def hook(self, sin, sout):
        h = self.heads
        q = self.to_q(sin[0])
        context = emb

        k = self.to_k(context)
        q=rearrange(q, '(b f) n (h d) -> (b h) (f n) d',f=f, h=h)
        k=rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim=rearrange(sim, 'b (f i) j -> b f i j',f=f)
        attn = sim.softmax(dim=-1)
        ret["out"] = attn
    return hook


def decode_attention_map(attn_map, video, output_mode,idx='', separate=False):
    output=video.clone().to("cpu").numpy()


    if (idx == ""):
        vid = attn_map[:,:,:,1:].sum(-1).sum(0).unsqueeze(0)
    else:
        try:
            idxs = list(map(int, filter(lambda x : x != '', idx.strip().split(','))))
            if separate:

                vid = rearrange(attn_map[:,:,:,idxs].sum(0), "f x n-> n f x")
                output=np.repeat(output,vid.shape[0],axis=0)
            else:
                vid = attn_map[:,:,:,idxs].sum(-1).sum(0).unsqueeze(0)
        except:
            return output

    scale = round(math.sqrt((video.shape[3] * video.shape[4]) / vid.shape[2]))
    n=vid.shape[0]
    f=video.shape[2]
    h = video.shape[3] // scale
    w = video.shape[4] // scale
    vid = vid.reshape(n,f,h, w) 
    for i in range(n):
        for j in range(f):
        
            vid[i][j]=vid[i][j]/vid[i][j].max()
        
    vid = vid.to("cpu").numpy()
    output = output.astype(np.float64)
    if output_mode == "masked":
        for i in range(output.shape[3]):
            for j in range(output.shape[4]):
                output[:,:,:,i,j] *= vid[:,:,i // scale,j // scale]

    elif output_mode == "grey":

        for i in range(output.shape[3]):
            for j in range(output.shape[4]):
                output[:,:,:,i,j] = np.repeat(np.expand_dims(vid[:,:,i // scale,j // scale],axis=1),3,axis=1)   
    return output

def layer_save_name(layer):
    if layer.startswith("mid"):
        return "mid"
    else:
        return f"{layer.split('_')[0]}_{layer.split('.')[1]}"
    
def video_residual(video):
    video=video.astype(np.float64)
    residual = video[1:]-video[:-1]
    residual[:,:,:,0]=np.where(residual[:,:,:,0]>=0,residual[:,:,:,0],0)
    residual[:,:,:,1]=np.where(residual[:,:,:,1]<0,-residual[:,:,:,1],0)
    residual[:,:,:,2]=np.where(residual[:,:,:,2]<0,0,0)
    residual=residual.astype(np.uint8)
    return residual

    

def video2longImage(video):
    image = video[0]
    for i in range(1,video.shape[0]):
        image = np.concatenate((image,video[i]),axis=1)
    return image

def get_idx(prompt):
    question=f"give the key nouns in the sentence that depict an entity in the scene. Only entities that will appear in the scene, not other nouns, should be given. Your response should only contain a list of nouns separated by commas. [Example: Sentence:'a blurry image of two people standing next to each other in a dark room'. Response: people, room] Now we have the Sentence: '{prompt}' "
    response = askChatGPT(question)
    if response is None:
        return None,None
    target_nouns=[noun.strip('\n').strip('.')for noun in response.split(", ")]
    if isinstance(prompt,list):
        prompt=prompt[0]
    prompt_list=word_tokenize(prompt)
    prompt_list=[i.lower() for i in prompt_list]
    idxs=[]
    nouns_not_found=[]
    print(target_nouns)
    for noun in target_nouns:
        try:
            idx=prompt_list.index(noun.lower())+1
            if idx>=76:
                continue
            idxs.append(str(idx))
        except Exception:
            nouns_not_found.append(noun)
            print(f"noun {noun} not in prompt")
    for noun in nouns_not_found:
        target_nouns.remove(noun)
            
        
    


    return ",".join(idxs),target_nouns


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet  # This is a no op
        # unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet",torch_dtype=torch.float16, variant="fp16")
        unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet")
        hidden_layers,hidden_layer_names=update_layer_names(unet)



    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    
        # torch_dtype=torch.float16, variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.unet.set_adapter

    unet.disable_gradient_checkpointing()
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()

    inject_inferable_lora(pipe, lora_path, r=lora_rank)

    return pipe, hidden_layer_names, hidden_layers


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        # initialize with random gaussian noise
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        # encode init_video to latents
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()


def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    latents: Tensor,
    init_weight: float,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    prompt_embeds: Optional[List[Tensor]],
    negative_prompt_embeds: Optional[List[Tensor]],
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
    rotate: bool,
    layer_selected: Optional[List[int]] = None,
    timestep_att: Optional[int] = None,
    layer_smooth: Optional[str] = None,
):
    device = pipe.device
    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size, _, num_frames, _, _ = latents.shape
    window_size = min(num_frames, window_size)

    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )
    neg_prompt_embeds = pipe._encode_prompt(
        prompt=[''],
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )
    prompt_embeds_without_neg = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # set the scheduler to start at the correct timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_step = round(init_weight * len(pipe.scheduler.timesteps))
    timesteps = pipe.scheduler.timesteps[start_step:]
    if init_weight == 0:
        latents = torch.randn_like(latents)
    else:
        latents = pipe.scheduler.add_noise(
            original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
        )

    # manually track previous outputs for the scheduler as we continually change the section of video being diffused
    model_outputs = [None] * order

    if rotate:
        shifts = np.random.permutation(primes_up_to(window_size))
        total_shift = 0
    attn_out={}
    if negative_prompt is not None:
        attn_neg_out={}
    else:
        attn_neg_out=None

    smooth=False
    if layer_smooth is not None:
        smooth=True

    with pipe.progress_bar(total=len(timesteps) * num_frames // window_size) as progress:
        for i, t in enumerate(timesteps):
            progress.set_description(f"Diffusing timestep {t}...")
        
            if rotate:  # rotate latents by a random amount (so each timestep has different chunk borders)
                shift = shifts[i % len(shifts)]
                model_outputs = [None if pl is None else torch.roll(pl, shifts=shift, dims=2) for pl in model_outputs]
                latents = torch.roll(latents, shifts=shift, dims=2)
                total_shift += shift

            new_latents = torch.zeros_like(latents)
            new_outputs = torch.zeros_like(latents)

            for idx in range(0, num_frames, window_size):  # diffuse each chunk individually

                Layer_Smooth=layer_smooth.copy() if layer_smooth is not None else None
                # update scheduler's previous outputs from our own cache
                pipe.scheduler.model_outputs = [model_outputs[(i - 1 - o) % order] for o in reversed(range(order))]
                pipe.scheduler.model_outputs = [
                    None if mo is None else mo[:, :, idx : idx + window_size, :, :].to(device)
                    for mo in pipe.scheduler.model_outputs
                ]
                pipe.scheduler.lower_order_nums = min(i, order)

                latents_window = latents[:, :, idx : idx + window_size, :, :].to(pipe.device)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_window] * 2) if do_classifier_free_guidance else latents_window
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                if layer_selected is not None and timestep_att==i:
                    if isinstance(layer_selected,list):
                        attn_out=[]
                        handles=[]
                        attn_neg_out=[]
                        handles_neg=[]
                        for layer in layer_selected:
                            a={}
                            handle = layer.register_forward_hook(get_attn(prompt_embeds, a,f=num_frames))
                            attn_out.append(a)
                            handles.append(handle)
                        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, layer_smooth=Layer_Smooth, smooth=smooth, smooth_function=attention_map_deflicker).sample

                        for handle in handles:
                            handle.remove()
                        if negative_prompt is not None:
                            for layer in layer_selected:
                                a_neg={}
                                handle_neg = layer.register_forward_hook(get_attn(neg_prompt_embeds, a_neg,f=num_frames))
                                attn_neg_out.append(a_neg)
                                handles_neg.append(handle_neg)
                            noise_pred_neg = pipe.unet(latent_model_input, t, encoder_hidden_states=neg_prompt_embeds).sample

                            for handle_neg in handles_neg:
                                handle_neg.remove()
                    else:
                        handle = layer_selected.register_forward_hook(get_attn(prompt_embeds, attn_out,f=num_frames))
                        if negative_prompt is not None:
                            handle_neg=layer_selected.register_forward_hook(get_attn(neg_prompt_embeds, attn_neg_out,f=num_frames))
                        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,layer_smooth=Layer_Smooth, smooth=smooth, smooth_function=attention_map_deflicker).sample
                        handle.remove()
                        if negative_prompt is not None:
                            handle_neg=layer_selected.register_forward_hook(get_attn(neg_prompt_embeds, attn_neg_out,f=num_frames))
                            noise_pred_neg = pipe.unet(latent_model_input, t, encoder_hidden_states=neg_prompt_embeds).sample
                            handle_neg.remove()
                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,layer_smooth=Layer_Smooth, smooth=smooth, smooth_function=attention_map_deflicker).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    #Perp-Neg
                    # if negative_prompt is not None:
                    #     proj_neg =noise_pred_uncond-torch.mul(torch.div(einsum("b c f h w, b c f h w -> b f", noise_pred_text, noise_pred_uncond),einsum("b c f h w, b c f h w -> b f", noise_pred_text, noise_pred_text)),noise_pred_text)
                    #     noise_pred = proj_neg + guidance_scale * (noise_pred_text - proj_neg)


                # reshape latents for scheduler
                pipe.scheduler.model_outputs = [
                    None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                    for mo in pipe.scheduler.model_outputs
                ]
                latents_window = rearrange(latents_window, "b c f h w -> (b f) c h w")
                noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

                # compute the previous noisy sample x_t -> x_t-1
                latents_window = pipe.scheduler.step(noise_pred, t, latents_window).prev_sample

                # reshape latents back for UNet
                latents_window = rearrange(latents_window, "(b f) c h w -> b c f h w", b=batch_size)

                # write diffused latents to output
                new_latents[:, :, idx : idx + window_size, :, :] = latents_window.cpu()

                # store scheduler's internal output representation in our cache
                new_outputs[:, :, idx : idx + window_size, :, :] = rearrange(
                    pipe.scheduler.model_outputs[-1], "(b f) c h w -> b c f h w", b=batch_size
                )

                progress.update()

            # update our cache with the further denoised latents
            latents = new_latents
            model_outputs[i % order] = new_outputs

    if rotate:
        new_latents = torch.roll(new_latents, shifts=-total_shift, dims=2)
    
    if layer_selected is not None:
        if isinstance(layer_selected,list):
            return new_latents,[a["out"] for a in attn_out], [a["out"] for a in attn_neg_out]
        else:
            return new_latents,attn_out["out"], attn_neg_out["out"]
    return new_latents,None


@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    window_size: Optional[int] = None,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    init_video: Optional[str] = None,
    init_weight: float = 0.5,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    loop: bool = False,
    seed: Optional[int] = None,
    layer_selected: Optional[str] = None,
    timestep_att: Optional[int] = None,
    output_mode: str = "grey",
    separate: bool = False,
    layer_smooth: Optional[List[int]] = None,
):
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(123)

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        pipe, hidden_layer_names, hidden_layers = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank)
        print(hidden_layer_names)
        print(len(hidden_layer_names))
        if layer_selected is not None:
            if isinstance(layer_selected,list):
                layer=[]
                for l in layer_selected:
                    layer.append(hidden_layers[l])
            else:
                layer=hidden_layers[layer_selected]

        # prepare prompts
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds, negative_prompt_embeds = compel(prompt), compel(negative_prompt) if negative_prompt else None

        # prepare input latents
        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            init_video=init_video,
            vae_batch_size=vae_batch_size,
        )
        init_weight = init_weight if init_video is not None else 0  # ignore init_weight as there is no init_video!
        # run diffusion

        latents, att_map, att_neg_map = diffuse(
            pipe=pipe,
            latents=init_latents,
            init_weight=init_weight,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            rotate=loop or window_size < num_frames,
            layer_selected=layer,
            timestep_att=timestep_att,
            layer_smooth=layer_smooth,
        )

        # decode latents to pixel space
        videos = decode(pipe, latents, vae_batch_size)
        idx_nouns,target_nouns=get_idx(prompt)
        if negative_prompt is not None:
            # if isinstance(att_neg_map,list):
            #     for i in range(len(att_neg_map)):
            #         att_neg_map[i]=decode_attention_map(att_neg_map[i],videos,output_mode,idx='')


            # else:
            #     att_neg_map=decode_attention_map(att_neg_map,videos,output_mode,idx='')
            if isinstance(att_map,list):
                for i in range(len(att_map)):
                    att_neg_map[i]=decode_attention_map(np.array_split(att_map[i],2)[0],videos,output_mode,idx='')


            else:
                att_neg_map=decode_attention_map(np.array_split(att_map,2)[0],videos,output_mode,idx='')
        if isinstance(att_map,list):
            for i in range(len(att_map)):
                if layer_selected[i]=="up_blocks.2.attentions.2.transformer_blocks.0.attn2" or layer_selected[i]=="up_blocks.1.attentions.2.transformer_blocks.0.attn2":
                    # att_map[i]=attention_map_deflicker(att_map[i],h=height,w=width)
                    pass
                att_map[i]=decode_attention_map(np.array_split(att_map[i],2)[1],videos,output_mode,idx="",separate=separate)

                # att_map[i]=decode_attention_map(att_map[i],videos,"grey",idx=idx_nouns)

        else:
            att_map=decode_attention_map(np.array_split(att_map,2)[1],videos,output_mode,idx="",separate=separate)
            # att_map=decode_attention_map(att_map,videos,"grey",idx=idx_nouns)
        
            
            
        


    return videos, att_map, att_neg_map,target_nouns


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Directory to save output video to")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=256, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=256, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-WS", "--window-size", type=int, default=None, help="Number of frames to process at once (defaults to full sequence). When less than num_frames, a round robin diffusion process is used to denoise the full sequence iteratively one window at a time. Must be divide num_frames exactly!")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=25, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-i", "--init-video", type=str, default=None, help="Path to video to initialize diffusion from (will be resized to the specified num_frames, height, and width).")
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5, help="Strength of visual effect of init_video on the output (lower values adhere more closely to the text prompt, but have a less recognizable init_video).")
    parser.add_argument("-f", "--fps", type=int, default=24, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-lP", "--lora_path", type=str, default="", help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lR", "--lora_rank", type=int, default=64, help="Size of the LoRA checkpoint's projection matrix (defaults to 64).")
    parser.add_argument("-rw", "--remove-watermark", action="store_true", help="Post-process the videos with LAMA to inpaint ModelScope's common watermarks.")
    parser.add_argument("-l", "--loop", action="store_true", help="Make the video loop (by rotating frame order during diffusion).")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-att", "--layer_selected", type=str, default=None, help="Which attention map to watch.")
    parser.add_argument("-t", "--timestep_att", type=int, default=None, help="Diffusion timestep of output attention map.")
    parser.add_argument("-om", "--output_mode", type=str, default="grey", help="output mode of attention maps")
    parser.add_argument("-sep", "--separate", type=bool, default=False, help="separate attention maps for different nouns")
    parser.add_argument("-smo", "--layer_smooth",nargs='+',default=None, help="which layer of attention map to smooth")
    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    out_name = f"{args.output_dir}/"
    if args.init_video is not None:
        out_name += f"[({Path(args.init_video).stem}) x {args.init_weight}] "
    prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", args.prompt) if platform.system() == "Windows" else args.prompt
    out_name += f"{prompt}"

    args.prompt = [prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size

    if args.window_size is None:
        args.window_size = args.num_frames

    if args.init_video is not None:
        vr = decord.VideoReader(args.init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args.num_frames, args.height, args.width), mode="trilinear")
        args.init_video = init

    # =========================================
    # ============= sample videos =============
    # =========================================
    default_hidden_layer_name = "mid_block.attentions.0.transformer_blocks.0.attn2"
    all_hidden_layer_name=["down_blocks.0.attentions.1.transformer_blocks.0.attn2",
                           "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
                           "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
                           "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
                           "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
                           "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
                           "mid_block.attentions.0.transformer_blocks.0.attn2"
                           ]
    if args.layer_selected=="all":
        args.layer_selected=all_hidden_layer_name
    if args.layer_smooth is not None and not isinstance(args.layer_smooth,list):
        args.layer_smooth=[args.layer_smooth]
    

    videos,att_map, attn_neg_map, target_nouns = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        window_size=args.window_size,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        init_video=args.init_video,
        init_weight=args.init_weight,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        loop=args.loop,
        layer_selected=args.layer_selected,
        timestep_att=args.timestep_att,
        output_mode=args.output_mode,
        separate=args.separate,
        layer_smooth=args.layer_smooth
    )
    target_nouns=['all']

    # =========================================
    # ========= write outputs to file =========
    # =========================================

    os.makedirs(args.output_dir, exist_ok=True)

    for i,video in enumerate(videos):
        if args.remove_watermark:
            print("Inpainting watermarks...")
            video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
            video = inpaint_watermark(video)
            video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

        else:
            video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

        video = video.byte().cpu().numpy()
        if isinstance(att_map,list):
            att=[]
            for a in att_map:
                if args.separate:
                    temp=(np.clip(rearrange(a[i::args.batch_size], "n c f h w -> n f h w c"),0,1))*255
                else:
                    temp=(np.clip(rearrange(a[i], "c f h w -> f h w c"),0,1))*255
                att.append(temp.astype(np.uint8))
        else:
            if args.separate:
                att=(np.clip(rearrange(att_map[i::args.batch_size], "n c f h w -> n f h w c"),0,1))*255
            else:
                att=(np.clip(rearrange(att_map[i], "c f h w -> f h w c"),0,1))*255
            att=att.astype(np.uint8)
        if isinstance(attn_neg_map,list):
            att_neg=[]
            for a in attn_neg_map:
                
                temp=(np.clip(rearrange(a[i], "c f h w -> f h w c"),0,1))*255
                att_neg.append(temp.astype(np.uint8))
        else:
            att_neg=(np.clip(rearrange(attn_neg_map[i], "c f h w -> f h w c"),0,1))*255
            att_neg=att_neg.astype(np.uint8)


        # export_to_video(video, f"{out_name} {str(uuid4())[:8]}.mp4", args.fps)
        if len(out_name)>100:
            out_name = out_name[:100]
            out_name+=f'_{args.timestep_att}'
        out_name+='deflicker'
        imageio.mimsave(f"{out_name}.gif", video,'GIF', fps=args.fps)
        if isinstance(att,list) or args.separate:
            os.makedirs(out_name, exist_ok=True)
            if args.separate:
                for i,at in enumerate(att):
                    save_name=layer_save_name(args.layer_selected[i])
                    for n,a in enumerate(at):
                        dir_name=f"{out_name}/{target_nouns[n]}"
                        os.makedirs(dir_name, exist_ok=True)
                        imageio.mimsave(f"{dir_name}/{save_name}.gif", a,'GIF', fps=args.fps)
                        imageio.mimsave(f"{dir_name}/{save_name}_residual.gif", video_residual(a),'GIF', fps=args.fps)
                        imageio.imwrite(f"{dir_name}/{save_name}.jpg",video2longImage(a))
                        imageio.imwrite(f"{dir_name}/{save_name}_residual.jpg",video2longImage(video_residual(a)))

                    
            else:
                for i,a in enumerate(att):
                    save_name=layer_save_name(args.layer_selected[i])
                    imageio.mimsave(f"{out_name}/{save_name}.gif", a,'GIF', fps=args.fps)
                    imageio.mimsave(f"{out_name}/{save_name}_residual.gif", video_residual(a),'GIF', fps=args.fps)
                    imageio.imwrite(f"{out_name}/{save_name}.jpg",video2longImage(a))
                    imageio.imwrite(f"{out_name}/{save_name}_residual.jpg",video2longImage(video_residual(a)))
            dir_name=f"{out_name}/negative"
            os.makedirs(dir_name, exist_ok=True)
            for i,a in enumerate(att_neg):
                save_name=layer_save_name(args.layer_selected[i])
                imageio.mimsave(f"{dir_name}/{save_name}.gif", a,'GIF', fps=args.fps)
                imageio.mimsave(f"{dir_name}/{save_name}_residual.gif", video_residual(a),'GIF', fps=args.fps)
                imageio.imwrite(f"{dir_name}/{save_name}.jpg",video2longImage(a))
                imageio.imwrite(f"{dir_name}/{save_name}_residual.jpg",video2longImage(video_residual(a)))
        else:
            if args.separate:
                for n,a in enumerate(att):
                    save_name=target_nouns[n]
                    imageio.mimsave(f"{out_name}/{save_name}.gif", a,'GIF', fps=args.fps)
                    imageio.mimsave(f"{out_name}/{save_name}_residual.gif", video_residual(a),'GIF', fps=args.fps)
                    imageio.imwrite(f"{out_name}/{save_name}.jpg",video2longImage(a))
                    imageio.imwrite(f"{out_name}/{save_name}_residual.jpg",video2longImage(video_residual(a)))
            else:
                imageio.mimsave(f"{out_name}_att_map.gif", att,'GIF', fps=args.fps)
                imageio.mimsave(f"{out_name}_att_map_residual.gif", video_residual(att),'GIF', fps=args.fps)
                imageio.imwrite(f"{out_name}_att_map.jpg",video2longImage(att))
                imageio.imwrite(f"{out_name}_att_map_residual.jpg",video2longImage(video_residual(att)))

