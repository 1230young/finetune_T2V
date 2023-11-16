import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("/raid/pengyuyang/code/Text-To-Video-Finetuning/models/potat1", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
prompt = "Iron man riding a horse."
video_frames = pipe(prompt, num_inference_steps=25, num_frames=20).frames

# convent to video
video_path = export_to_video(video_frames,"test.mp4")