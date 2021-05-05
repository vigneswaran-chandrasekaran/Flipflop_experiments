import tensorflow as tf
import random
import re
import os
import tempfile
import cv2
import numpy as np
import imageio
import numpy as np

def crop_center_square(frame):
  #We have to make changes here
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('animation.gif', converted_images, fps=25)

categories = {}
for video in ucf_videos:
  category = video[2:-12]
  #print(category)
  if category not in categories:
    categories[category] = []
  categories[category].append(video)
print("Found %d videos in %d categories." % (len(ucf_videos), len(categories)))

for category, sequences in categories.items():
  summary = ", ".join(sequences[:2])
  print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))

def create_labels_for_videos(video_list, classes=101): #ucf_101
    c = 0
    video_dict = {}
    no_of_videos = len(video_list)
    labels = []
    for i in range(no_of_videos):
        name = video_list[i]
        first_name = name[2:-12]
        
        if first_name not in video_dict:
            video_dict[first_name] = c
            c += 1
        
        labels.append(video_dict[first_name])
    #labels = to_categorical(labels, classes)
    return video_dict, labels


class DataLoader():

  def __init__(self, batch_size=4):

    _, labels = create_labels_for_videos(ucf_videos)
    video_list = videos_lister
    labels = np.array(labels)
    videos = np.array(video_list)
    idx = np.random.permutation(labels.shape[0])

    self.labels = labels[idx]
    self.videos = videos[idx]

    self.idx = 0
    self.batch_size = batch_size
    self.finished = False

  def spit_data(self):

    batch_x = []
    batch_y = []

    offset = self.batch_size + self.idx
    if self.idx + self.batch_size > self.labels.shape[0]:
        offset = self.labels.shape[0]
        self.finished = True
  
    for i in range(self.idx, offset):
      v_i = self.videos[i]
      batch_x.append(load_video(fetch_ucf_video(v_i))[:50])
      batch_y.append(self.labels[i])

    self.idx += self.batch_size

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    return batch_x, batch_y
        