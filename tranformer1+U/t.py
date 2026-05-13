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

# Transformer 模块
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, dim_feedforward=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.flatten_size = channels  # 对于 U-Net，channels 可以视为 token 的数量
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, channels))  # 位置编码
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channels, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
            ),
            num_layers=1  # 可以根据需要增加 Transformer 层数
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width)  # 展平为 [batch_size, channels, H*W]

        # 动态调整位置编码大小以匹配展平后的维度
        pos_embedding = self.pos_embedding[:, :, :x.size(2)]  # 确保位置编码大小与 H*W 一致
        x = x + pos_embedding

        x = x.permute(2, 0, 1)  # 转换为 [sequence_len, batch_size, channels]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # 转换回 [batch_size, channels, sequence_len]
        x = x.view(batch_size, channels, height, width)  # 恢复为 [batch_size, channels, H, W]
        return x

# 修改后的 U-Net SNN 模型，包含 Transformer 模块
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, T=16, *args, **kwargs):
        super().__init__()

        self.T = T  # 时间步数

        # 下采样路径（编码器）
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = 2  # 输入通道数，DVS128Gesture 的数据通道为2
        for i in range(5):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                neuron.LIFNode(*args, **kwargs)
            )
            self.encoders.append(conv)
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = channels  # 更新输入通道数

        # TransformerBlock 放在编码器和解码器之间的瓶颈位置
        self.transformer = TransformerBlock(channels, num_heads=8)

        # 上采样路径（解码器）
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(5):
            upconv = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                neuron.LIFNode(*args, **kwargs)
            )
            self.decoders.append(conv)

        # 最后的输出层，添加了 AdaptiveAvgPool2d
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            neuron.LIFNode(*args, **kwargs),
            nn.AdaptiveAvgPool2d((4, 4)),  # 添加自适应平均池化层
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels * 4 * 4, 512),
            neuron.LIFNode(*args, **kwargs),
            nn.Dropout(0.5),
            nn.Linear(512, 11),  # 输出类别数为11
        )

    def forward(self, x: torch.Tensor):
        # x 的形状为 [N, T, C, H, W]
        N, T, C, H, W = x.shape

        # 初始化脉冲神经元的状态
        functional.reset_net(self)

        outputs = []

        for t in range(self.T):
            xt = x[:, t, :, :, :]  # 当前时间步的输入，形状为 [N, C, H, W]
            # 编码器部分
            enc_outs = []
            out = xt
            for i in range(5):
                out = self.encoders[i](out)
                enc_outs.append(out)
                out = self.pools[i](out)

            # TransformerBlock 在编码器的瓶颈位置加入
            out = self.transformer(out)

            # 解码器部分
            for i in range(5):
                out = self.upconvs[i](out)
                enc_out = enc_outs[4 - i]
                # 尺寸对齐
                diffH = enc_out.size(2) - out.size(2)
                diffW = enc_out.size(3) - out.size(3)
                out = F.pad(out, [diffW // 2, diffW - diffW // 2,
                                  diffH // 2, diffH - diffH // 2])
                # 拼接
                out = torch.cat([out, enc_out], dim=1)
                out = self.decoders[i](out)

            out = self.final_conv(out)  # 输出形状为 [N, num_classes]
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)  # 形状为 [T, N, num_classes]
        outputs = outputs.mean(0)  # 在时间维度上求平均，得到 [N, num_classes]

        return outputs

def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture with U-Net SNN and Transformer')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda', help='device')
    parser.add_argument('-b', default=4, type=int, help='batch size')
    parser.add_argument('-epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='/home/dell/171990/datasets/DVS128Gesture', help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='/home/dell/171990/SNNs train/out H1/tau50', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, default='/home/dell/171990/SNNs train/out H1/tau50/T16_b4_adam_lr0.0006_c64_unet/checkpoint_max.pth', help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.0006, type=float, help='learning rate')
    parser.add_argument('-channels', default=64, type=int, help='channels of CSNN')

    args = parser.parse_args()
    print(args)

    net = DVSGestureNet(channels=args.channels, T=args.T, surrogate_function=surrogate.ATan(), detach_reset=True)

    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    net.to(args.device)

    print(net)

    # 加载数据集
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        print(f"Invalid optimizer type: {args.opt}")
        raise NotImplementedError(args.opt)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_unet')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)  # 形状为 [N, T, C, H, W]
            label = label.to(args.device)

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame)
                    loss = F.cross_entropy(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame)
                loss = F.cross_entropy(out_fr, label)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)  # 形状为 [N, T, C, H, W]
                label = label.to(args.device)
                out_fr = net(frame)
                loss = F.cross_entropy(out_fr, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'estimated finish time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

if __name__ == '__main__':
    main()
