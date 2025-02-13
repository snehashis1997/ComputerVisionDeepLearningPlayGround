import cv2
import imageio
import os
from glob import glob

img_list = glob("/workspaces/Workfiles/siren/visualization/*.png")
img_list.sort(reverse=False)

frames = []

for i in range(0,1000,20):

    item = os.path.join(f"/workspaces/Workfiles/siren/visualization/denoised_{i}.png")
    frame = cv2.imread(item)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)

# Save frames as a GIF
gif_path = "output.gif"
imageio.mimsave(gif_path, frames, fps=5)

print(f"GIF saved at {gif_path}")