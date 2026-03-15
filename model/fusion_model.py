# model/fusion_model.py
import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x):
        # x: [B,T,C,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x).view(B*T, -1)
        return self.fc(feats).view(B, T, -1)


class SCADAEncoder(nn.Module):
    def __init__(self, in_dim=10, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FusionModel(nn.Module):
    def __init__(self, scada_dim=10):
        super().__init__()
        self.video_enc = VideoEncoder()
        self.scada_enc = SCADAEncoder(scada_dim)

        self.fusion = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.temporal = nn.LSTM(128, 128, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [cascade_prob, failure_time, voltage]
        )

    def forward(self, video, scada):
        """
        video: [B,T,3,H,W]
        scada: [B,T,scada_dim]
        """
        v_feat = self.video_enc(video)     # [B,T,128]
        s_feat = self.scada_enc(scada)     # [B,T,128]

        fused, _ = self.fusion(v_feat, s_feat, s_feat)
        temporal_out, _ = self.temporal(fused)

        out = self.head(temporal_out[:, -1])
        return out