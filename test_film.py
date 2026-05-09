import torch
from deepsv.models.fused_cnn import FusedDeepSV

model = FusedDeepSV(embed_dim=256, num_classes=2, input_channels=3)
image = torch.randn(2, 3, 255, 255)
embedding = torch.randn(2, 256)
out = model(image, embedding)
print("Output shape:", out.shape)
print("Success! FiLM injection works.")
