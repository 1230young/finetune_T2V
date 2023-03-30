import os
import decord
import numpy as np
import random
import json
import torchvision.transforms as T
import torch
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from glob import glob

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    if use_caption:
        caption_file = ''
        # Use caption on per-video basis (One caption PER video)
        for ext in ext_types:
            maybe_file = file_path.replace(ext, '.txt')
            if maybe_file.endswith(ext_types): continue
            if os.path.exists(maybe_file): 
                caption_file = maybe_file
                break

        if os.path.exists(caption_file):
            read_caption_file(caption_file)
        
        # Return text prompt if no conditions are met.
        return text_prompt

    return text_prompt

def path_or_prompt(caption_path, prompt):
    if os.path.exists(self.single_caption_path):
        prompt = read_caption_file(caption_path)
    else:
        return prompt
    
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            base_width: int = 256,
            base_height: int = 256,
            n_sample_frames: int = 4,
            n_sample_frames_min: int = 1,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            json_path: str ="./data",
            vid_data_key: str = "video_path",
            use_random_start_idx: bool = False,
            preprocessed: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")

        self.tokenizer = tokenizer
        self.preprocessed = preprocessed

        self.train_data = self.load_from_json(json_path)
        self.vid_data_key = vid_data_key

        self.original_start_idx = sample_start_idx
        self.use_random_start_idx = use_random_start_idx

        self.width = width
        self.height = height

        self.base_width = base_width
        self.base_height = base_height
        self.use_base = False

        self.n_sample_frames = n_sample_frames
        self.n_sample_frames_min = n_sample_frames_min
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.sample_frame_rate_init = sample_frame_rate

    def load_from_json(self, path):
        try:
            print(f"Loading JSON from {path}")
            with open(path) as jpath:
                json_data = json.load(jpath)
            
            if not self.preprocessed:
                for data in json_data['data']:
                    is_valid = self.validate_json(json_data['base_path'],data["folder"])
                    if not is_valid:
                        raise ValueError(f"{data['folder']} is not a valid folder for path {json_data['base_path']}.")

                print(f"{json_data['name']} successfully loaded.")
                return json_data
            else: 
                print("Preprocessed mode.")
                return json_data

        except:
            raise ValueError("Invalid JSON")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, idx, vr):
        frames = self.get_sample_frame()
        return list(range(idx, len(vr), self.sample_frame_rate))[:frames]
    
    def get_sample_idx(self, idx, vr):
        # Get the frame idx range based on the get_vid_idx function
        # We have a fallback here just in case we the frame cannot be read
        sample_idx = self.get_frame_range(idx, vr)
        fallback = self.get_frame_range(1, vr)
        
        # Return the result from the get_vid_idx function. This will error out if it cannot be read.
        try:
            vr.get_batch(sample_idx)
            return sample_idx

        # Return the fallback frame range if it fails
        except:
            return fallback

    def get_sample_frame(self):
        return self.n_sample_frames

    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames

        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = 1

        return idx

    def get_width_height(self):
        if self.use_base: return self.base_width, self.base_width
        else: return self.width, self.height

    def train_data_batch(self, index):
         # Assign train data
        train_data = self.train_data['data'][index]

        # load and sample video frames

        width, height = self.get_width_height()
        vr = decord.VideoReader(train_data[self.vid_data_key], width=width, height=height)

        # Pick a random video from the dataset
        vid_data = random.choice(train_data['data'])

        # Set a variable framerate between 1 and 30 FPS 
        idx = self.get_vid_idx(vr, vid_data)

        # Check if idx is greater than the length of the video.
        if idx >= len(vr):
            idx = 1
            
        # Resolve sample index
        sample_index = self.get_sample_idx(idx, vr)
        
        # Get video prompt
        prompt = vid_data['prompt']

        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        # Video JSON
        if self.train_data is not None:
            return len(self.train_data['data'])

        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        
        # Single Video
        return 1

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt
        }

        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            base_width: int = 256,
            base_height: int = 256,
            n_sample_frames: int = 4,
            sample_frame_rate: int = 1,
            use_random_start_idx: bool = False,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            single_caption_path: str = "",
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.use_random_start_idx = use_random_start_idx

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt
        self.single_caption_path = single_caption_path

        self.width = width
        self.height = height
        self.curr_video = None
        self.sample_frame_rate = sample_frame_rate
        self.sample_frame_rate_init = sample_frame_rate

    def get_sample_frame(self):
        return self.n_sample_frames

    def get_vid_idx(self, vr, vid_data=None):
        frames = self.get_sample_frame()
        if self.use_random_start_idx:
            
            # Randomize the frame rate at different speeds
            self.sample_frame_rate = random.randint(1, self.sample_frame_rate_init)

            # Randomize start frame so that we can train over multiple parts of the video
            random.seed()
            max_sample_rate = abs((frames - self.sample_frame_rate) + 2)
            max_frame = abs(len(vr) - max_sample_rate)
            idx = random.randint(1, max_frame)
            
        else:
            if vid_data is not None:
                idx = vid_data['frame_index']
            else:
                idx = 1

        return idx

    def get_frame_range(self, idx, vr):
        frames = self.get_sample_frame()
        return list(range(idx, len(vr), self.sample_frame_rate))[:frames]
    
    def get_sample_idx(self, idx, vr):
        # Get the frame idx range based on the get_vid_idx function
        # We have a fallback here just in case we the frame cannot be read
        sample_idx = self.get_frame_range(idx, vr)
        fallback = self.get_frame_range(1, vr)
        
        # Return the result from the get_vid_idx function. This will error out if it cannot be read.
        try:
            vr.get_batch(sample_idx)
            return sample_idx

        # Return the fallback frame range if it fails
        except:
            return fallback

    def single_video_batch(self):
        train_data = self.single_video_path
        if train_data.endswith(self.vid_types):

            if self.curr_video is None:
                self.curr_video = decord.VideoReader(train_data, width=self.width, height=self.height)
            # Load and sample video frames
            vr = self.curr_video

            idx = self.get_vid_idx(vr)

            # Check if idx is greater than the length of the video.
            if idx >= len(vr):
                idx = 1
                
            # Resolve sample index
            sample_index = self.get_sample_idx(idx, vr)

            # Process video and rearrange
            video = vr.get_batch(sample_index)
            video = rearrange(video, "f h w c -> f c h w")


        prompt = path_or_prompt(self.single_caption_path, self.single_video_prompt)
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        if os.path.exists(self.single_video_path): return 1
        return 0

    def __getitem__(self, index):
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        video, prompt, prompt_ids = self.single_video_batch()

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt
        }

        return example
    
class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption: bool = False,
        image_dir: str = '',
        single_caption_path: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.image_dir = self.get_images_list(image_dir)

        self.use_caption = use_caption
        self.single_caption_path = single_caption_path

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        width = self.width
        height = self.height

        if img.shape[2] > img.shape[1]:
            height = abs(self.height - 192)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)

        prompt = get_text_prompt(
            file_path=train_data,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Do image training
        if os.path.exists(self.image_dir[0]):
            img, prompt, prompt_ids = self.image_batch(index)

        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt
        }

        return example

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        **kwargs
    ):
        self.tokenizer = tokenizer

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        vr = decord.VideoReader(self.video_files[index], width=self.width, height=self.height)
        native_fps = vr.get_avg_fps()
        every_nth_frame = round(native_fps / self.fps)

        effective_length = len(vr) // every_nth_frame

        if effective_length < self.n_sample_frames:
            return self.__getitem__(random.randint(0, len(self.video_files) - 1))

        effective_idx = random.randint(0, effective_length - self.n_sample_frames)

        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + self.n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        else:
            prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt}
