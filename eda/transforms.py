# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tt

i_p = "/vol/miltank/projects/practical_WS2425/diffusion/code/eda/0001-1000_orig.png"
i_p = "/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_ct_all_axis/multi/img_gen/3_2_pred.png"

i = Image.open(i_p)

plt.axis("off")
plt.imshow(i)

# %%
ts = tt.Compose(
    [
        tt.ToTensor(),
        tt.ColorJitter(brightness=0, contrast=0.5, saturation=0),
    ]
)


i_t = ts(i)

plt.axis("off")
plt.imshow(i_t.permute(1, 2, 0))
