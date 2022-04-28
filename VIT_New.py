import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            a = x[0]
            return a, self.fn(self.norm(x[1]), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.norm(self.net(x))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return attn, self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            ai = attn(x)
            x = ai[1] + x
            x = ff(x) + x

        return ai[0], self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, device, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.channels = channels
        self.dim = dim
        self.device = device
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token3 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token4 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token5 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token6 = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.pos_embedding1 = nn.Parameter(torch.randn(1, 65, self.dim)).to(self.device)
        self.pos_embedding2 = nn.Parameter(torch.randn(1, 10, self.dim)).to(self.device)
        self.pos_embedding3 = nn.Parameter(torch.randn(1, 10, self.dim)).to(self.device)
        self.pos_embedding4 = nn.Parameter(torch.randn(1, 10, self.dim)).to(self.device)
        self.pos_embedding5 = nn.Parameter(torch.randn(1, 10, self.dim)).to(self.device)
        self.pos_embedding6 = nn.Parameter(torch.randn(1, 10, self.dim)).to(self.device)

        # self.conv1 = nn.Sequential(nn.Conv2d(3, 10, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                           nn.Conv2d(10, 100, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                           nn.Conv2d(100, 3, 2, padding=1), nn.MaxPool2d(2, stride=1))
        #
        # self.conv2 = nn.Sequential(nn.Conv2d(3, 10, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(10, 100, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(100, 3, 2, padding=1), nn.MaxPool2d(2, stride=1))
        # self.conv3 = nn.Sequential(nn.Conv2d(3, 10, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(10, 100, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(100, 3, 2, padding=1), nn.MaxPool2d(2, stride=1))
        # self.conv4 = nn.Sequential(nn.Conv2d(3, 10, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(10, 100, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(100, 3, 2, padding=1), nn.MaxPool2d(2, stride=1))
        # self.conv5 = nn.Sequential(nn.Conv2d(3, 10, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(10, 100, 2, padding=1), nn.MaxPool2d(2, stride=1),
        #                            nn.Conv2d(100, 3, 2, padding=1), nn.MaxPool2d(2, stride=1))
        self.transformers = nn.ModuleList()
        self.cnn1 = nn.Conv2d(3, self.dim, kernel_size=128, stride=64)
        self.cnn2 = nn.Conv2d(3, self.dim, kernel_size=64, stride=32)
        self.cnn3 = nn.Conv2d(3, self.dim, kernel_size=32, stride=16)
        self.cnn4 = nn.Conv2d(3, self.dim, kernel_size=16, stride=8)
        self.cnn5 = nn.Conv2d(3, self.dim, kernel_size=8, stride=4)
        self.linear = nn.Linear(3, self.dim)
        self.transformers.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        #self.transformers.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        #self.transformers.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.transformers1 = nn.ModuleList()
        self.transformers1.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers1.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers1.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.transformers2 = nn.ModuleList()
        self.transformers2.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers2.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers2.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.transformers3 = nn.ModuleList()
        self.transformers3.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers3.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers3.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.transformers4 = nn.ModuleList()
        self.transformers4.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers4.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers4.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.transformers5 = nn.ModuleList()
        self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))
        # self.transformers5.append(Transformer(dim, depth, heads, dim_head, mlp_dim, dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(6144),
            nn.Linear(6144, num_classes)
        )

    def extract_patch(self, img):
        image_height, image_width = img.size()[2:]
        # patch_height = int(image_height / 4)
        # patch_width = int(image_width / 4)
        # patch_dim = self.channels * patch_height * patch_width
        # if patch_height == 64:
        #     embed = self.linears1
        # if patch_height == 32:
        #     embed = self.linears2
        # if patch_height == 16:
        #     embed = self.linears3
        # if patch_height == 8:
        #     embed = self.linears5
        # if patch_height == 4:
        #     embed = self.linears6
        # ra = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        # x_new = embed(x_new)
        if image_height == 256:
            embed = self.cnn1
        if image_height == 128:
            embed = self.cnn2
        if image_height == 64:
            embed = self.cnn3
        if image_height == 32:
            embed = self.cnn4
        if image_height == 16:
            embed = self.cnn5

        x = embed(img)
        x_new = torch.zeros((img.size()[0], 9, self.dim)).to(self.device)
        x_new[:, 0, :] = x[:, :, 0, 0]
        x_new[:, 1, :] = x[:, :, 0, 1]
        x_new[:, 2, :] = x[:, :, 0, 2]
        x_new[:, 3, :] = x[:, :, 1, 0]
        x_new[:, 4, :] = x[:, :, 1, 1]
        x_new[:, 5, :] = x[:, :, 1, 2]
        x_new[:, 6, :] = x[:, :, 2, 0]
        x_new[:, 7, :] = x[:, :, 2, 1]
        x_new[:, 8, :] = x[:, :, 2, 2]

        return x_new

    def patch_embedding(self, img):

        x = self.extract_patch(img)
        b, n, _ = x.shape
        image_height = img.size()[3]
        if image_height == 256:
            pos = self.pos_embedding2
            cls_tokens = repeat(self.cls_token1, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if image_height == 128:
            pos = self.pos_embedding3
            cls_tokens = repeat(self.cls_token2, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if image_height == 64:
            pos = self.pos_embedding4
            cls_tokens = repeat(self.cls_token3, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if image_height == 32:
            pos = self.pos_embedding5
            cls_tokens = repeat(self.cls_token4, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        if image_height == 16:
            pos = self.pos_embedding6
            cls_tokens = repeat(self.cls_token5, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x += pos[:, :(n + 1)]
        return x

    def extract_batch_from_img(self, img, idx):
        b, c, h, w = img.size()
        batches = []
        for r, idxx in enumerate(idx):
            if idxx == 0:
                im = img[r, :, 0:int(h / 2), 0:int(w / 2)]
            if idxx == 1:
                im = img[r, :, int(h / 4): int(3 * h / 4), 0:int(w / 2)]
            if idxx == 2:
                im = img[r, :, int(h / 2): h, 0:int(w / 2)]
            if idxx == 3:
                im = img[r, :, 0:int(h / 2), int(w / 4):int(3 * w / 4)]
            if idxx == 4:
                im = img[r, :, int(h / 4): int(3 * h / 4), int(w / 4):int(3 * w / 4)]
            if idxx == 5:
                im = img[r, :, int(h / 2): h, int(w / 4):int(3 * w / 4)]
            if idxx == 6:
                im = img[r, :, 0:int(h / 2), int(w / 2):w]
            if idxx == 7:
                im = img[r, :, int(h / 4): int(3 * h / 4), int(w / 2):w]
            if idxx == 8:
                im = img[r, :, int(h / 2): h, int(w / 2):w]
            batches.append(im)
        return torch.stack(batches, dim=0)

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        for layer in self.transformers:
            att, x = layer(x)
        cls_1 = torch.clone(x[:, 0])
        b, h, n, _ = att.shape

        best_patch = torch.argmax(torch.sum(torch.mean(att[:, :, 1:n, 1:n], dim=1), dim=1), dim=1)
        new_img = self.extract_batch_from_img(img, best_patch)
        x = self.patch_embedding(new_img)
        x = self.dropout(x)
        for layer in self.transformers1:
            att, x = layer(x)
        cls_2 = torch.clone(x[:, 0])
        b, h, n, _ = att.shape
        best_patch = torch.argmax(torch.sum(torch.mean(att[:, :, 1:n, 1:n], dim=1), dim=1), dim=1)
        new_img = self.extract_batch_from_img(new_img, best_patch)

        x = self.patch_embedding(new_img)
        x = self.dropout(x)
        for layer in self.transformers2:
            att, x = layer(x)
        cls_3 = torch.clone(x[:, 0])
        b, h, n, _ = att.shape
        best_patch = torch.argmax(torch.sum(torch.mean(att[:, :, 1:n, 1:n], dim=1), dim=1), dim=1)
        new_img = self.extract_batch_from_img(new_img, best_patch)
        #
        x = self.patch_embedding(new_img)
        x = self.dropout(x)
        for layer in self.transformers3:
            att, x = layer(x)
        cls_4 = torch.clone(x[:, 0])
        b, h, n, _ = att.shape
        best_patch = torch.argmax(torch.sum(torch.mean(att[:, :, 1:n, 1:n], dim=1), dim=1), dim=1)
        new_img = self.extract_batch_from_img(new_img, best_patch)

        x = self.patch_embedding(new_img)
        x = self.dropout(x)
        for layer in self.transformers4:
            att, x = layer(x)
        cls_5 = torch.clone(x[:, 0])
        b, h, n, _ = att.shape
        best_patch = torch.argmax(torch.sum(torch.mean(att[:, :, 1:n, 1:n], dim=1), dim=1), dim=1)
        new_img = self.extract_batch_from_img(new_img, best_patch)

        x = new_img
        ra = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        x = ra(x)
        embed = self.linear

        x = embed(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token6, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding1[:, :(n + 1)]
        x = self.dropout(x)
        for layer in self.transformers5:
            att, x = layer(x)
        cls_6 = x[:, 0]
        x1 = self.to_latent(cls_1)
        x2 = self.to_latent(cls_2)
        x3 = self.to_latent(cls_3)
        x4 = self.to_latent(cls_4)
        x5 = self.to_latent(cls_5)
        x6 = self.to_latent(cls_6)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        return self.mlp_head(x)
