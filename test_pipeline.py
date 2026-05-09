import torch
import torch.nn as nn
from deepsv.models.cnn import ModernDeletionCNN
from deepsv.models.fused_cnn import FusedDeepSV

# Test M0 (CNN only)
cnn = ModernDeletionCNN(num_classes=2)
x = torch.randn(2, 3, 256, 256)
out_cnn = cnn(x)
print("CNN output shape:", out_cnn.shape)

# Test M1 (Fused)
fused = FusedDeepSV(embed_dim=256, num_classes=2)
emb = torch.randn(2, 256)
out_fused = fused(x, emb)
print("Fused output shape:", out_fused.shape)

print("Pipeline integration successful!")
