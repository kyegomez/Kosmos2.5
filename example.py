import torch
from kosmos.model import Kosmos

#text
text = torch.randint(0, 256, (1, 1024)).cuda()
img = torch.randn(1, 3, 256, 256)

#multimodal GPT4

model = Kosmos()
model = Kosmos(text, img)