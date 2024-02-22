import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
#from openmixup.models.augments import cutmix




class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            #print(self.train_loader)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    # def mixup_data(x, y, alpha=1.0, use_cuda=True):
    #     lam = np.random.beta(alpha, alpha)
    #     batch_size = x.size()[0]
    #     if use_cuda:
    #         index = torch.randperm(batch_size).cuda()
    #     else:
    #         index = torch.randperm(batch_size)

    #     mixed_x = lam * x + (1 - lam) * x[index, :]
    #     y_a, y_b = y, y[index]
    #     return mixed_x, y_a, y_b, lam

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        if self.config['arch']['args']['freeze_bn'] and epoch > 15:
        #if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        # print("=============================================================")
        # print(tbar)
        # print("=============================================================")
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)
            #print(target.size())
            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            #print(output.size()[2:])
            #print(target.size()[1:])
            #if self.config['arch']['type'][:3] == 'PSP':
            if self.config['arch']['type'][:4] == 'FCPF':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes 
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, target)
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            
            
            if batch_idx % self.log_step == 0:
                #self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.wrt_step = epoch
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            mpa, pixAcc, mIoU, Precision, Recall, _ = self._get_seg_metrics().values()
            #Precision, Recall, FNR = self._getprecision().values()
            #print("Precision:",Precision)
            #print("Recall:",Recall)
            #print("FNR:",FNR)
            
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} |mPa {:.2f} Acc {:.2f} mIoU {:.2f} Prec.{:.2f} Rec.{:.2f}| B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average, 
                                                mpa, pixAcc, mIoU, Precision, Recall,
                                                self.batch_time.average, self.data_time.average))
            #Precision, Recall, FNR = self._getprecision().values()
            #print("Precision:",Precision)



        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                
                if len(val_visual) < 400:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                
                mpa, pixAcc, mIoU, Precision, Recall, _ = self._get_seg_metrics().values()
                #Precision, Recall, FNR = self._getprecision().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, MPixAcc: {:.2f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, Precision: {:.2f} , Recall: {:.2f} |' .format(epoch,
                                                self.total_loss.average,
                                                mpa, pixAcc, mIoU, Precision, Recall))
                #mpa, pixAcc, mIoU, _ = self._get_seg_metrics().values()
                #tbar.set_description('EVAL ({}) | Loss: {:.3f}, MPixAcc: {:.2f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |' .format( epoch,
                #                                self.total_loss.average,
                #                                mpa, pixAcc, mIoU))
                ###替身
                #Precision, Recall, FNR = self._getprecision().values()
                #tbar.set_description('EVAL ({}) | Loss: {:.3f}, MPixAcc: {:.2f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, Precision: {:.2f} , Recall: {:.2f} , FNR: {:.2f} |' .format(epoch,
                #                                self.total_loss.average,
                #                                mpa, pixAcc, mIoU, Precision, Recall, FNR))
                #print("Precision:",Precision)
            
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            
            #self.wrt_step = (epoch) * len(self.val_loader)
           
            self.wrt_step = epoch
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.total_a, self.total_b = 0, 0
        self.total_FN, self.total_FP, self.total_TN = 0, 0 ,0
    def _update_seg_metrics(self, correct, labeled, inter, union, a, b, FN, FP, target):
    #def _update_seg_metrics(self, correct, labeled, inter, union, a, b):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_a += a
        self.total_b += b
        #self.correct = correct
        #self.labeled = labeled
        self.total_FN += FN
        self.total_FP += FP
        self.total_TN += (target-inter-FN-FP)
    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)           ##total_inter=TP,total_union=FN+FP+TP
        pa = 1.0 * self.total_a / (np.spacing(1) + self.total_b)
        mIoU = IoU.mean()
        mpa = pa.mean()
        #pixelacc_2 = 1.0 * self.correct / (np.spacing(1) + self.labeled)
        #Mean_pixelacc_2 = pixelacc_2.mean()
        Precision = 1.0 * self.total_inter/(np.spacing(1)+self.total_inter+self.total_FP)
        Recall = 1.0 * self.total_inter / (np.spacing(1) + self.total_inter + self.total_FN)
        #FNR = 1.0 * self.total_FN / (np.spacing(1) + self.total_inter + self.total_FN)
        #FPR = 1.0 * self.total_FP / (np.spacing(1) + self.total_TN + self.total_FP)
        Precision = Precision.mean()
        Recall = Recall.mean()
        #FNR = FNR.mean()
        #FPR = FPR.mean()
        with open('log.txt', 'w', encoding='utf-8') as f:
            f.writelines("Pixel_Accuracy:"+str(np.round(pixAcc, 3))+'\n')
            f.writelines("MeanPixel_Accuracy:" + str(np.round(mpa, 3)) + '\n')
            f.writelines("Mean_IoU:"+ str(np.round(mIoU, 3)) + '\n')
            f.writelines("Precision:"+ str(np.round(Precision, 3)) + '\n')
            f.writelines("Recall:"+ str(np.round(Recall, 3)) + '\n')
            #f.writelines("FNR:"+ str(np.round(FNR, 3)) + '\n')
            #f.writelines("FNR:"+ str(np.round(FPR, 3)) + '\n')
            #f.writelines("MeanPixel_Accuracy_2:" + str(np.round(Mean_pixelacc_2, 3)) + '\n')
            f.writelines("Class_IoU:"+ str(dict(zip(range(self.num_classes), np.round(IoU, 3)))) + '\n')

        return {
            "MeanPixel_Accuracy": np.round(mpa, 3),
            #"MeanPixel_Accuracy_2": np.round(Mean_pixelacc_2, 3),
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Precision": np.round(Precision, 3),
            "Recall": np.round(Recall, 3),
            #"FNR": np.round(FNR, 3),
            #"FPR": np.round(FPR, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    #def _getprecision(self):
        #Precision = 1.0 * self.total_inter/(np.spacing(1)+self.total_inter+self.total_FP)
        #Recall = 1.0 * self.total_inter / (np.spacing(1) + self.total_inter + self.total_FN)
        #FNR = 1.0 * self.total_FN / (np.spacing(1) + self.total_inter + self.total_FN)
        #Precision = Precision.mean()
        #Recall = Recall.mean()
        #FNR = FNR.mean()
        #FPR = 1.0 * self.total_FP / (np.spacing(1) + self.total_TN + self.total_FP)

     #   return {
     #       "Precision": np.round(Precision, 3),
     #       "Recall": np.round(Recall, 3),
     #       "FNR": np.round(FNR, 3)
     #   }