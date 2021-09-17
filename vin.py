import torch
import torch.nn as nn


class ViN(nn.Module):
    def __init__(self, img_size, num_patches, embedding_dropout, num_classes, dim, depth, mlp_dim):
        super(ViN, self).__init__()

        patch_size = int(img_size / num_patches)
        self.rearrange = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pose_embedding = nn.Parameter(torch.randn(1, num_patches**2+1, dim))
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout, inplace=True)
        self.transformer = nn.TransformerDecoderLayer(dim, depth, mlp_dim, activation='gelu')
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x = self.rearrange(x)
        # x = self.linear_proj(x)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat((self.cls_tokens.repeat(b, 1, 1), x), dim=1)
        x += self.pose_embedding.repeat(b, 1, 1)
        self.embedding_dropout(x)
        x = self.transformer(x, x)
        x = self.mlp_head(x[:, 0])
        return x


if __name__ == '__main__':
    net = ViN(512, 4, 0.1, 500, 768, 12, 3072)
    torch.save(net.state_dict(), './aa')
    x = torch.randn(10, 3, 512 ,512)
    net.to('cuda:0')
    x = x.to('cuda:0')
    y=net(x)
    print(y)