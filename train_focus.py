import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import datetime
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from dataloaders import make_data_loader
from dm.create_network import DyNet
from utils.loss.compound_losses import DC_and_CE_loss
from utils.loss.compound_losses import MemoryEfficientSoftDiceLoss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
import matplotlib.pyplot as plt
import pickle
from utils.save_feature_map import save_map_1,save_map_2,save_map_img
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device=torch.device("cuda")
        self.args.checkname="tprnet" 
        self.saver = Saver(args)
        self.experiment_dir=self.saver.experiment_dir
        self.global_output_dir=os.path.join(self.experiment_dir,'img/global_output/')
        self.global_local_output_dir=os.path.join(self.experiment_dir,'img/global_local_output/')
        self.final_output_dir=os.path.join(self.experiment_dir,'img/final_output/')
        self.focus_output_dir=os.path.join(self.experiment_dir,'img/focus_output/')
        self.pkl_dir=os.path.join(self.experiment_dir,'img/pkl/')
        if not os.path.exists(self.global_output_dir):
            os.makedirs(self.global_output_dir)
        if not os.path.exists(self.global_local_output_dir):
            os.makedirs(self.global_local_output_dir)
        if not os.path.exists(self.final_output_dir):
            os.makedirs(self.final_output_dir)
        if not os.path.exists(self.focus_output_dir):
            os.makedirs(self.focus_output_dir)
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
        if args.cuda:
            self.device=torch.device('cuda')
        self.saver.save_experiment_config()
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args)
        model=DyNet(square_size=args.square_size,local_layer=args.local_layer,patch_size=args.patch_size,focus_layer=args.focus_layer,step=args.step,centerline_ratio=args.centerline_ratio)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)
        self.scheduler = LR_Scheduler(optimizer, args.lr, args.epochs)
        self.loss = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,dice_class=MemoryEfficientSoftDiceLoss)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator()
        self.grad_scaler=GradScaler()
        self.train_losss=[]
        self.val_losss=[]
        self.train_dices=[]
        self.val_dices=[]
        if args.cuda:
            self.model = self.model.cuda()
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['network_weights'])
            else:
                self.model.load_state_dict(checkpoint['network_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    def croptarget(self,target,start,length):
        target=target[:,:,start[0]:start[0]+length,start[1]:start[1]+length]
        return target
    def batch_loss(self, sub_output_with_coord, target):
        total_loss = 0
        num_samples = len(sub_output_with_coord)
        for cropped_img, (top_left, bottom_right) in sub_output_with_coord:
            cropped_target = target[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            loss = self.loss(cropped_img, cropped_target)
            total_loss += loss.item()
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss
    def fusion(self, output,sub_output_with_coord):#array,(array,tuple)
        fused_output = output.clone()
        for cropped_img, (top_left, bottom_right) in sub_output_with_coord:
            cropped_output = fused_output[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            alpha = self.args.focus_ratio
            #下策：调整权重使得fused_region倾向更好的结果
            #中策：使cropped_img的结果尽可能好
            fused_region = alpha * cropped_img + (1 - alpha) * cropped_output
            fused_output[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = fused_region
        return fused_output
    def training(self, epoch):
        train_loss = 0.0
        self.evaluator.reset()
        self.model.train()
        current_lr=self.scheduler.step(epoch)
        print('\n=>Epoches %i, learning rate = %.10f, previous best = %.4f' % (epoch, current_lr, self.best_pred))
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=True):
                if epoch>0:
                    output1,output2,output3,start,length = self.model(image,epoch)
                    target2=self.croptarget(target,start,length)
                    output12=output1.clone()
                    output12[:,:,start[0]:start[0]+length,start[1]:start[1]+length]=(output12[:,:,start[0]:start[0]+length,start[1]:start[1]+length]+output2)/2
                    output=self.fusion(output12,output3)
                    #loss=0.6*(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output12, target)+self.loss(output, target)/4)+0.4*self.batch_loss(output3,target)
                    loss=(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output12, target)+self.loss(output, target)+self.batch_loss(output3,target))/5
                else:
                    output1,output2,start,length = self.model(image,epoch)
                    target2=self.croptarget(target,start,length)
                    output=output1.clone()
                    output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]=(output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]+output2)/2
                    loss=(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output, target))/3
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            train_loss += loss.item()
            avg_train_loss=train_loss / (i + 1)
            tbar.set_description('train loss: %.3f' % (avg_train_loss))
            
            pred=F.softmax(output,1)[:,1]
            pred=pred.data
            pred[pred>0.5]=1
            pred[pred<=0.5]=0
            self.evaluator.add_batch(target[:,0], pred)
        self.train_losss.append(avg_train_loss)
        dice=self.evaluator.Dice_coefficient()
        self.train_dices.append(dice)
    def save_loss_plot(self):
        epochs = range(1, len(self.train_losss) + 1)
        train_losses = self.train_losss
        val_losses = self.val_losss
        train_dices = self.train_dices
        val_dices = self.val_dices
    
        max_val_dice_value = max(val_dices)
        max_val_dice_index = val_dices.index(max_val_dice_value)

        plt.switch_backend('Agg')
    
        plt.figure(figsize=(16, 6))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue', lw=1.5)
        ax1.plot(epochs, val_losses, label='Val Loss', color='green', lw=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(epochs, train_dices, label='Train Dice', color='blue', lw=1.5)
        ax2.plot(epochs, val_dices, label='Val Dice', color='green', lw=1.5)
        ax2.scatter(max_val_dice_index + 1, max_val_dice_value, color='green', s=50, zorder=5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.set_title('Dice')
        ax2.legend()
        ax2.grid(True)
    
        plot_filename = os.path.join(self.saver.experiment_dir, str(max_val_dice_value)+'loss_dice.png')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=100)
        plt.close()
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        val_outputs = []
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            if epoch>0:
                output1,output2,output3,start,length = self.model(image,epoch)
                target2=self.croptarget(target,start,length)
                output12=output1.clone()
                output12[:,:,start[0]:start[0]+length,start[1]:start[1]+length]=(output12[:,:,start[0]:start[0]+length,start[1]:start[1]+length]+output2)/2
                output=self.fusion(output12,output3)
                
                loss=(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output12, target)+self.loss(output, target)+self.batch_loss(output3,target))/5
                #loss=0.6*(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output12, target)+self.loss(output, target)/4)+0.4*self.batch_loss(output3,target)
            else:
                output1,output2,start,length = self.model(image,epoch)
                target2=self.croptarget(target,start,length)
                output=output1.clone()
                output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]=(output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]+output2)/2
                
                loss=(self.loss(output1, target)+self.loss(output2, target2)+self.loss(output, target))/3
            val_loss += loss.item()
            avg_test_loss=val_loss / (i + 1)
            tbar.set_description('val loss: %.3f' % (avg_test_loss))
            pred=F.softmax(output,1)[:,1]
            pred[pred>0.5]=1
            pred[pred<=0.5]=0
            self.evaluator.add_batch(target[:,0], pred)
        dice=self.evaluator.Dice_coefficient()
        self.val_dices.append(dice)
        self.val_losss.append(avg_test_loss)
        if epoch==self.args.epochs-1:
            self.save_loss_plot()
        print("dice:%.4f"%dice)
        new_pred = dice
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.best_epoch=epoch
            self.saver.save_checkpoint({
               'epoch': epoch + 1,
               'network_weights': self.model.state_dict(),
               'optimizer_state': self.optimizer.state_dict(),
               'best_pred': self.best_pred,
            }, is_best)
    def fusion_show(self, sub_output_with_coord):
        fused_output = torch.zeros(1, 2, 512, 512).to(self.device)
        
        for cropped_img, (top_left, bottom_right) in sub_output_with_coord:
            # 获取当前区域的切片
            cropped_output = fused_output[:, :, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # 找到fused_output中该区域为0的位置（未被替代的区域）
            mask_zero = (cropped_output == 0)
            
            # 对于为0的部分，直接替代为cropped_img
            cropped_output[mask_zero] = cropped_img[mask_zero]
            
            # 对于已经替代过的部分（非0），进行简单平均
            mask_non_zero = (cropped_output != 0)
            cropped_output[mask_non_zero] = (cropped_output[mask_non_zero] + cropped_img[mask_non_zero]) / 2

            # 将更新后的部分回写到fused_output
            fused_output[:, :, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cropped_output

        return fused_output
    def save_coord(self,sub_output_with_coord,name):
        coord=[]
        for _, (top_left, bottom_right) in sub_output_with_coord:
            coord.append((top_left, bottom_right))
        with open(self.pkl_dir+name+'.pkl', 'wb') as f:
            pickle.dump(coord, f)
    def test(self):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                output1, output2, output3, start, length = self.model(image, 1)
                target2 = self.croptarget(target, start, length)
                output12 = output1.clone()
                output12[:, :, start[0]:start[0]+length, start[1]:start[1]+length] = (output12[:, :, start[0]:start[0]+length, start[1]:start[1]+length] + output2) / 2
                output = self.fusion(output12, output3)
                sub_output=self.fusion_show(output3)
                self.save_coord(output3,sample[1][0])

            pred = torch.softmax(output, dim=1)[:,1]
            save_map_1(pred,sample[1][0],self.final_output_dir)
            pred1 = torch.softmax(output1, dim=1)[:,1]
            save_map_1(pred1,sample[1][0],self.global_output_dir)
            pred12 = torch.softmax(output12, dim=1)[:,1]
            save_map_1(pred12,sample[1][0],self.global_local_output_dir)
            pred3 = torch.softmax(sub_output, dim=1)[:,1]
            save_map_1(pred3,sample[1][0],self.focus_output_dir)
            pred[pred>0.5]=1
            pred[pred<=0.5]=0
            self.evaluator.add_batch(target[:, 0], pred)

        dice = self.evaluator.Dice_coefficient()
        print(f"Dice coefficient: {dice:.4f}")
        return dice    

def main(patch_size,focus_layer,centerline_ratio,focus_ratio):
    
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--dataset', type=str, default='arcade',
                        help='dataset name (default: arcade)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--square_size', type=int, default=320, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--patch_size', type=int, default=320, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--step', type=int, default=320, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--ratio', type=int, default=320, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--local_layers', type=int, default=320, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--tr_batch-size', type=int, default=8, 
                        metavar='N', help='input batch size for \
                                training (default: 16)') 
    parser.add_argument('--val_batch-size', type=int, default=8, 
                        metavar='N', help='input batch size for \
                                training (default: 16)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.99,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-5,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--runname', type=str, default='run',
                        help='set the checkpoint name')
    parser.add_argument('--checkname', type=str, default='global',
                        help='set the checkpoint name')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    args = parser.parse_args()
    args.tr_batch_size=1
    args.val_batch_size=1
    args.square_size=448
    args.patch_size=patch_size
    args.step=patch_size
    args.local_layer=7
    args.focus_layer=focus_layer
    args.centerline_ratio=centerline_ratio
    args.focus_ratio=focus_ratio
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        time0=datetime.datetime.now()
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            with torch.no_grad():
                trainer.validation(epoch)
        time1=datetime.datetime.now()
        print("spend:",((time1-time0).seconds),"s")
    trainer.test()

if __name__ == "__main__":
    r=[0.5]
    for i in r:
        main(64,4,0.5,i)
