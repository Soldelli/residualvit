import io
import math
import random

import torch

from PIL import Image
from decord import VideoReader

VIDEO_EXTENSIONS = ['.mp4']

class VideoDecoder(object):
    def __init__(self, num_frames=1, target_fps=None):
        self.num_frames = num_frames
        if target_fps==None or target_fps==0.0:
            self.decode = self.decode_random_frame_sequence_with_decord
        else:
            self.target_fps = target_fps
            self.decode = self.decode_random_frame_sequence_with_decord_at_fixed_fps
        
    def decode_random_frame_sequence_with_decord_at_fixed_fps(self, file_type, video_bytes):
        if file_type not in VIDEO_EXTENSIONS:
            return None
        vobj = io.BytesIO(video_bytes)
        
        target_fps = self.target_fps
        if target_fps == 0.0:
            target_fps = random.choice([0.5, 1.0, 2.0, 3.0])
        try:
            video = VideoReader(vobj)
            video_num_frames, video_fps = len(video),video.get_avg_fps()
            minimum_num_frames = math.ceil(self.num_frames * video_fps / self.target_fps)

            if video_num_frames < minimum_num_frames:
                # This if clause allows training with videos at a different target fps in case they are too short
                if video_num_frames < self.num_frames:
                    raise ValueError('Not enough frames in this video')
                
                step = video_num_frames // self.num_frames
                frame_idx = list(range(0, video_num_frames, step))
            else:
                # Let's sample the num_frames from a random starting frame aiming at respecting the target FPS.
                step = math.ceil(video_fps / self.target_fps)
                rnd_idx = random.choice(range(0, max(video_num_frames - minimum_num_frames, 1)))
                frame_idx = list(range(rnd_idx, rnd_idx + minimum_num_frames, step))

            frame_idx = frame_idx[:self.num_frames] # Trim to avoid any rounding issue

            assert len(frame_idx) == self.num_frames
            random_seq = [
                Image.fromarray(video[idx].asnumpy()).convert('RGB') for idx in frame_idx
            ]

        except Exception as e:
            random_seq = [Image.new('RGB', (256, 256)) for idx in range(self.num_frames)]
            print(f'LOADING FAILED FOR SOME REASON - {e}')

        return random_seq

    def decode_random_frame_sequence_with_decord(self, file_type, video_bytes):
        if file_type not in VIDEO_EXTENSIONS:
            return None
        vobj = io.BytesIO(video_bytes)
        try:
            video = VideoReader(vobj)
            video_fps = video.get_avg_fps()
            video_num_frames = len(video)
            rnd_idx = random.choice(range(video_num_frames-self.num_frames))
            random_seq = [
                Image.fromarray(video[idx].asnumpy()).convert('RGB') for idx in range(rnd_idx, rnd_idx+self.num_frames)
            ]
        except Exception as e:
            random_seq = [
                Image.new('RGB', (256, 256)) for idx in range(self.num_frames)
            ]
            print(f'LOADING FAILED FOR SOME REASON')
        return random_seq
    
    