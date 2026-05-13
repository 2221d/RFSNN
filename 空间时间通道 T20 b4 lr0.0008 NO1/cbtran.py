import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
import pandas as pd  # 用于保存Excel
import math  # 用于计算位置编码



# ---------------------- 原始模块定义（CBAM、PositionalEncoding、TemporalSelfAttention） ----------------------

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_out, avg_out], dim=1)
        sa = self.spatial_attention(sa)
        return x * sa


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, num_classes, num_heads=8, d_model=64, dropout=0.1):
        super(TemporalSelfAttention, self).__init__()
        self.input_proj = nn.Linear(num_classes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.attention = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [T, N, num_classes]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [N, T, d_model]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # [T, N, d_model]
        x = self.out_proj(attn_output)
        return x


# ---------------------- 主模型定义（DVSGestureNet） ----------------------

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, T=16, num_classes=11, *args, **kwargs):
        super(DVSGestureNet, self).__init__()
        self.T = T
        self.num_classes = num_classes

        # 下采样路径（编码器）
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = 2
        for _ in range(5):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(channels),
                neuron.LIFNode(*args, **kwargs)
            )
            self.encoders.append(conv)
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = channels

        # 空间注意力（CBAM）
        self.cbam = CBAM(channels)

        # 上采样路径（解码器）
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(5):
            up = nn.ConvTranspose2d(channels, channels,
                                    kernel_size=2, stride=2)
            self.upconvs.append(up)
            dec = nn.Sequential(
                nn.Conv2d(channels * 2, channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                neuron.LIFNode(*args, **kwargs)
            )
            self.decoders.append(dec)

        # 输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            neuron.LIFNode(*args, **kwargs),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels * 4 * 4, 512),
            neuron.LIFNode(*args, **kwargs),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # 改进时间注意力
        self.temporal_attention = TemporalSelfAttention(
            num_classes=num_classes, num_heads=8, d_model=64)

    def forward(self, x: torch.Tensor):
        # x: [N, T, C, H, W]
        N, T, C, H, W = x.shape
        functional.reset_net(self)
        outputs = []
        for t in range(self.T):
            out = x[:, t, :, :, :]
            enc_outs = []
            for i in range(5):
                out = self.encoders[i](out)
                enc_outs.append(out)
                out = self.pools[i](out)
            out = self.cbam(out)
            for i in range(5):
                out = self.upconvs[i](out)
                enc = enc_outs[4 - i]
                diffH = enc.size(2) - out.size(2)
                diffW = enc.size(3) - out.size(3)
                out = F.pad(out, [diffW // 2, diffW - diffW // 2,
                                  diffH // 2, diffH - diffH // 2])
                out = torch.cat([out, enc], dim=1)
                out = self.decoders[i](out)
            out = self.final_conv(out)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)  # [T, N, num_classes]
        outputs = self.temporal_attention(outputs)
        outputs = outputs.mean(0)  # [N, num_classes]
        return outputs


# ---------------------- Reptile-style 元学习函数 ----------------------

def reptile_meta_update(model, loss, meta_lr):
    """
    给定模型当前梯度（已 backward），
    用 Reptile 公式更新 model 参数：
    θ := θ + meta_lr * (θ_fast - θ_orig)
    通过勾子(hook)或手动实现，这里我们手动保存 orig、fast。
    """
    # (实现见 train loop 中的具体步骤)
    pass  # 该函数留空，实际更新在 train loop 中完成。


# ---------------------- 主函数 ----------------------

def main():
    parser = argparse.ArgumentParser(
        description='使用U-Net SNN + Reptile元学习 on DVS Gesture')
    parser.add_argument('-T', default=16, type=int,
                        help='仿真时间步数')
    parser.add_argument('-device', default='cuda',
                        help='训练设备')
    parser.add_argument('-b', default=4, type=int,
                        help='批大小')
    parser.add_argument('-epochs', default=400, type=int,
                        help='训练轮数')
    parser.add_argument('-j', default=4, type=int,
                        help='DataLoader 线程数')
    parser.add_argument('-data-dir', type=str,
                        default='D:/datasets/DVS128Gesture',
                        help='数据集根目录')
    parser.add_argument('-out-dir', type=str,
                        default='D:/output/cbam_reptile',
                        help='输出/日志目录')
    parser.add_argument('-resume', type=str, default='',
                        help='checkpoint 恢复路径')
    parser.add_argument('-amp', action='store_true',
                        help='是否使用自动混合精度')
    parser.add_argument('-cupy', action='store_true',
                        help='是否使用 cupy 后端')
    parser.add_argument('-opt', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='优化器类型')
    parser.add_argument('-momentum', default=0.9, type=float,
                        help='SGD 动量')
    parser.add_argument('-lr', default=0.0008, type=float,
                        help='学习率 (原训练流程)')
    parser.add_argument('-channels', default=64, type=int,
                        help='CSNN 通道数')
    # 元学习超参
    parser.add_argument('--inner-lr', default=1e-2, type=float,
                        help='内环学习率')
    parser.add_argument('--inner-steps', default=1, type=int,
                        help='内环更新步数')
    parser.add_argument('--meta-lr', default=1e-3, type=float,
                        help='元学习率')
    args = parser.parse_args()

    # 打印配置
    print(args)

    # 设备
    device = args.device

    # 模型
    net = DVSGestureNet(channels=args.channels, T=args.T,
                        surrogate_function=surrogate.ATan(),
                        detach_reset=True).to(device)
    if args.cupy:
        functional.set_backend(net, 'cupy',
                               instance=neuron.LIFNode)
    print(net)

    # 原始 optimizer & scheduler
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)

    # DataLoader
    train_set = DVS128Gesture(root=args.data_dir,
                              train=True,
                              data_type='frame',
                              frames_number=args.T,
                              split_by='number')
    test_set = DVS128Gesture(root=args.data_dir,
                             train=False,
                             data_type='frame',
                             frames_number=args.T,
                             split_by='number')
    train_loader = DataLoader(train_set,
                              batch_size=args.b,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.j,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.b,
                             shuffle=False,
                             drop_last=False,
                             num_workers=args.j,
                             pin_memory=True)

    # AMP
    scaler = amp.GradScaler() if args.amp else None

    # 输出目录 & TensorBoard
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(args.out_dir)
    with open(os.path.join(args.out_dir, 'args.txt'),
              'w', encoding='utf-8') as f:
        f.write(str(args) + '\n' + ' '.join(sys.argv))

    # 恢复
    start_epoch = 0
    max_test_acc = -1
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt['epoch'] + 1
        max_test_acc = ckpt.get('max_test_acc', -1)

    # 用于 Excel 保存
    metrics = []

    # ---------- 训练循环（含 Reptile 内环+元更新） ----------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            N = frames.size(0)

            # 1) 备份原始参数
            orig_params = {n: p.data.clone()
                           for n, p in net.named_parameters()}

            # 2) 内环多步 SGD on 同 batch
            inner_opt = torch.optim.SGD(net.parameters(),
                                       lr=args.inner_lr)
            for _ in range(args.inner_steps):
                inner_opt.zero_grad()
                out_inner = net(frames)
                loss_inner = F.cross_entropy(out_inner,
                                             labels)
                loss_inner.backward()
                inner_opt.step()

            # 3) 元更新（Reptile 插值）
            with torch.no_grad():
                for n, p in net.named_parameters():
                    fast = p.data
                    p.data = orig_params[n] + \
                             args.meta_lr * (fast - orig_params[n])

            # 4) 原训练流程
            optimizer.zero_grad()
            if scaler:
                with amp.autocast():
                    out = net(frames)
                    loss = F.cross_entropy(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = net(frames)
                loss = F.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()

            # 累计指标
            bs = labels.size(0)
            train_samples += bs
            train_loss += loss.item() * bs
            train_acc += (out.argmax(1) == labels).float().sum().item()

        # 训练指标归一化
        t1 = time.time()
        train_loss /= train_samples
        train_acc /= train_samples
        train_speed = train_samples / (t1 - t0)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        # ---------- 测试循环 ----------
        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        t2 = time.time()
        with torch.no_grad():
            for frames, labels in test_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                out = net(frames)
                loss = F.cross_entropy(out, labels)
                bs = labels.size(0)
                test_samples += bs
                test_loss += loss.item() * bs
                test_acc += (out.argmax(1) == labels).float().sum().item()
        t3 = time.time()
        test_loss /= test_samples
        test_acc /= test_samples
        test_speed = test_samples / (t3 - t2)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        # 保存 checkpoint
        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
        if save_max:
            torch.save(ckpt,
            os.path.join(args.out_dir, 'checkpoint_max.pth'))
        torch.save(ckpt,
        os.path.join(args.out_dir, 'checkpoint_latest.pth'))

        # 控制台打印
        eta = datetime.datetime.now() + \
              datetime.timedelta(
                  seconds=(t1 - t0) * (args.epochs - epoch)
              )
        print(args)
        print(args.out_dir)
        print(f'第 {epoch} 轮，训练损失: {train_loss:.4f}, '
              f'训练准确率: {train_acc:.4f}, '
              f'测试损失: {test_loss:.4f}, '
              f'测试准确率: {test_acc:.4f}, '
              f'最佳测试准确率: {max_test_acc:.4f}')
        print(f'训练速度: {train_speed:.2f} 张图片/秒，'
              f'测试速度: {test_speed:.2f} 张图片/秒')
        print(f'预计完成时间: {eta.strftime("%Y-%m-%d %H:%M:%S")}\n')

        # Excel 保存
        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'max_test_acc': max_test_acc
        })
        pd.DataFrame(metrics).to_excel(
            os.path.join(args.out_dir, 'metrics.xlsx'),
            index=False
        )

    writer.close()


if __name__ == '__main__':
    main()