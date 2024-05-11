from tqdm import tqdm
import os
import torch
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr
import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock, Bottleneck
import time
import copy
import sys
import gc
import pandas as pd

from torch.utils.data import DataLoader

from utils.util import setup_seed, set_logging, SaveOutput
from script.extract_feature import get_resnet_feature, get_vit_feature
from options.train_options import TrainOptions
from model.deform_regressor import deform_fusion, Pixel_Prediction
# from data.pipal import PIPAL
# from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, crop_image, Normalize, five_point_crop
# from torchvision import transforms


class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.regressor.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay},
            {'params': self.deform_net.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay}
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.eval_predictions_data = None

        if self.opt.test is True:
            self.load_model_from_path(self.opt.checkpoints_dir)
            if self.opt.evaluation_type == "traditional_datasets":
                # self.test(self.test_loader)
                self.test(self.val_loader)
            elif self.opt.evaluation_type == "cross_dataset":
                self.test_cross_dataset()
        else:
            self.load_model()
            self.train()

            if self.opt.evaluation_type == "cross_dataset":
                self.eval_cross_dataset()

    def eval_cross_dataset(self):
        logging.info('Starting cross-dataset evaluation...')
        for test_loader in self.test_loaders:
            logging.info(f"Running testing on {test_loader.dataset}...")
            self.eval_epoch(self.opt.n_epoch, test_loader)

    def test_cross_dataset(self):
        print('Starting cross-dataset testing...')
        for test_loader in self.test_loaders:
            print(f"Running testing on {test_loader.dataset}...")
            self.test(test_loader)

    def create_model(self):
        self.resnet50 = timm.create_model('resnet50', pretrained=True).cuda()
        if self.opt.patch_size == 8:
            self.vit = timm.create_model('vit_base_patch8_224', pretrained=True).cuda()
        else:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
        self.deform_net = deform_fusion(self.opt).cuda()
        self.regressor = Pixel_Prediction().cuda()

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def init_data(self):
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

        from lib_iqa.data.image_transformation import AHIQTransform
        from lib_iqa.data.datasets import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, PIPALDataset

        ROOT_DIR = os.path.join("/", "opt", "simonra", "perceptual_metric")
        DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
        LIVE_DS_PATH = os.path.join(DATASETS_DIR, "live")
        CSIQ_DS_PATH = os.path.join(DATASETS_DIR, "csiq")
        TID2013_DS_PATH = os.path.join(DATASETS_DIR, "tid2013")
        KADID10K_DS_PATH = os.path.join(DATASETS_DIR, "kadid10k")
        PIPAL_DS_PATH = os.path.join(DATASETS_DIR, "pipal")
        # train_dataset = PIPAL(
        #     ref_path=self.opt.train_ref_path,
        #     dis_path=self.opt.train_dis_path,
        #     txt_file_name=self.opt.train_list,
        #     transform=transforms.Compose(
        #         [
        #             RandCrop(self.opt.crop_size, self.opt.num_crop),
        #             #Normalize(0.5, 0.5),
        #             RandHorizontalFlip(),
        #             ToTensor(),
        #         ]
        #     ),
        # )
        # val_dataset = PIPAL(
        #     ref_path=self.opt.val_ref_path,
        #     dis_path=self.opt.val_dis_path,
        #     txt_file_name=self.opt.val_list,
        #     transform=ToTensor(),
        # )

        dl_kwargs = {
            "drop_last": True, "prefetch_factor": 2, "pin_memory": True, "persistent_workers": True,
        }

        if self.opt.evaluation_type == "traditional_datasets":
            transform = AHIQTransform(size=self.opt.crop_size, crop=True, eval_center_crop=self.opt.eval_center_crop)
            if self.opt.dataset == 'LIVE':
                dataset = LIVEDataset(root_dir=LIVE_DS_PATH, transform=copy.deepcopy(transform))
            elif self.opt.dataset == 'CSIQ':
                dataset = CSIQDataset(root_dir=CSIQ_DS_PATH, transform=copy.deepcopy(transform))
            elif self.opt.dataset == 'TID2013':
                dataset = TID2013Dataset(root_dir=TID2013_DS_PATH, transform=copy.deepcopy(transform))
            elif self.opt.dataset == 'KADID-10K':
                dataset = KADID10KDataset(root_dir=KADID10K_DS_PATH, transform=copy.deepcopy(transform))
            elif self.opt.dataset == 'PIPAL':
                dataset = PIPALDataset(root_dir=PIPAL_DS_PATH, transform=copy.deepcopy(transform))
            else:
                raise ValueError('Dataset not supported')

            sets_split = (0.8, 0.2, 0)  # train, val, test

            normalize_labels = True
            dataset.reset_labels(normalize_labels)
            train_indexes, val_indexes, test_indexes = dataset.get_sets_split_indexes(*sets_split, seed=self.opt.seed)
            if sets_split[2] == 0:
                val_indexes += test_indexes

            train_dataset = torch.utils.data.Subset(copy.deepcopy(dataset), train_indexes)
            val_dataset = torch.utils.data.Subset(copy.deepcopy(dataset), val_indexes)
            test_dataset = torch.utils.data.Subset(copy.deepcopy(dataset), test_indexes)

            train_dataset.dataset.split = "train_split"
            val_dataset.dataset.split = "val_split"
            test_dataset.dataset.split = "test_split"
            self.opt.val_batch_size = self.opt.batch_size
            val_dataset.dataset.is_train = test_dataset.dataset.is_train = False

            logging.info('number of train scenes: {}'.format(len(train_dataset)))
            logging.info('number of val scenes: {}'.format(len(val_dataset)))
            logging.info('number of test scenes: {}'.format(len(test_dataset)))
            print(f'Using dataset: {dataset}')

            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.opt.val_batch_size,
                num_workers=self.opt.num_workers,
                shuffle=False,
                **dl_kwargs
            )
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.opt.val_batch_size,
                num_workers=self.opt.num_workers,
                shuffle=False,
                **dl_kwargs
            )

        elif self.opt.evaluation_type == "cross_dataset":
            self.opt.val_batch_size = self.opt.batch_size
            if self.opt.dataset == 'KADID-10K':
                ds_path, ds_cls = KADID10K_DS_PATH, KADID10KDataset
            elif self.opt.dataset == 'PIPAL':
                ds_path, ds_cls = PIPAL_DS_PATH, PIPALDataset
            else:
                raise ValueError('Dataset not supported')

            train_dataset = ds_cls(root_dir=ds_path, transform=AHIQTransform(size=self.opt.crop_size, crop=True))
            test_datasets = [
                LIVEDataset(
                    root_dir=LIVE_DS_PATH, transform=AHIQTransform(size=self.opt.crop_size, crop=True), is_train=False),
                CSIQDataset(
                    root_dir=CSIQ_DS_PATH, transform=AHIQTransform(size=self.opt.crop_size, crop=True), is_train=False),
                TID2013Dataset(
                    root_dir=TID2013_DS_PATH, transform=AHIQTransform(size=self.opt.crop_size, crop=True), is_train=False)]

            self.test_loaders = [DataLoader(
                dataset=test_ds, batch_size=self.opt.val_batch_size,
                num_workers=self.opt.num_workers,
                shuffle=False, **dl_kwargs) for test_ds in test_datasets]

            train_dataset.split = "train_split"
            for test_ds in test_datasets:
                test_ds.split = f"train_{str(train_dataset)}_test_split"
        else:
            raise ValueError('Evaluation type not supported')

        if self.opt.evaluation_type == "cross_dataset" or sets_split[0] > 0:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                num_workers=self.opt.num_workers,
                batch_size=self.opt.batch_size,
                shuffle=True,
                **dl_kwargs
            )
        else:
            self.train_loader = None

    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        if os.path.exists(models_dir):
            if self.opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self.opt.load_epoch = load_epoch
                checkpoint = torch.load(os.path.join(models_dir, "epoch_"+str(self.opt.load_epoch)+".pth"))
                self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch']+1
                loss = checkpoint['loss']
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        found = int(file.split('.')[0].split('_')[1]) == self.opt.load_epoch
                        if found:
                            break

                    if not found:
                        print(f"Model for epoch {self.opt.load_epoch} not found")
                        self.opt.load_epoch = 0

                    # assert found, 'Model for epoch %i not found' % self.opt.load_epoch
        else:
            assert self.opt.load_epoch < 1, 'Model for epoch %i not found' % self.opt.load_epoch
            self.opt.load_epoch = 0

    def forward(self, data):
        # d_img_org = data['d_img_org'].cuda()
        # r_img_org = data['r_img_org'].cuda()
        # labels = data['score']
        d_img_org = data.distortion_image.cuda()
        r_img_org = data.reference_image.cuda()
        labels = data.label_normalized
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

        _x = self.vit(d_img_org)
        vit_dis = get_vit_feature(self.save_output)
        self.save_output.outputs.clear()

        _y = self.vit(r_img_org)
        vit_ref = get_vit_feature(self.save_output)
        self.save_output.outputs.clear()
        B, N, C = vit_ref.shape
        if self.opt.patch_size == 8:
            H, W = 28, 28
        else:
            H, W = 14, 14
        assert H*W == N
        vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
        vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

        _ = self.resnet50(d_img_org)
        cnn_dis = get_resnet_feature(self.save_output)  # 0,1,2都是[B,256,56,56]
        self.save_output.outputs.clear()
        cnn_dis = self.deform_net(cnn_dis, vit_ref)

        _ = self.resnet50(r_img_org)
        cnn_ref = get_resnet_feature(self.save_output)
        self.save_output.outputs.clear()
        cnn_ref = self.deform_net(cnn_ref, vit_ref)

        pred = self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

        return pred, labels

    def train_epoch(self, epoch):
        losses = []
        self.regressor.train()
        self.deform_net.train()
        self.vit.eval()
        self.resnet50.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(self.train_loader):
            pred, labels = self.forward(data)
            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
        logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

        return ret_loss, rho_s, rho_p

    def train(self):
        best_srocc = 0
        best_plcc = 0
        self.get_total_model_params()
        self.get_total_params_size()
        for epoch in range(self.opt.load_epoch, self.opt.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = self.train_epoch(epoch)
            if self.opt.evaluation_type == "traditional_datasets" and (epoch + 1) % self.opt.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))

                loss, rho_s, rho_p = self.eval_epoch(epoch, self.val_loader)
                logging.info('Eval done...')

                if rho_s > best_srocc or rho_p > best_plcc:
                    best_srocc = rho_s
                    best_plcc = rho_p
                    print('Best now')
                    logging.info('Best now')
                    self.save_model(epoch, "best.pth", loss, rho_s, rho_p)
                    if epoch % self.opt.save_interval == 0:
                        weights_file_name = "epoch_%d.pth" % (epoch+1)
                        self.save_model(epoch, weights_file_name, loss, rho_s, rho_p)

            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

        # Save last model for evluation:
        if self.opt.evaluation_type == "cross_dataset":
            weights_file_name = "last.pth"
            self.save_model(epoch, weights_file_name, loss_val, rho_s, rho_p)

        # Tear down the dataloaders to free up memory
        self.train_loader = None
        gc.collect()
        torch.cuda.empty_cache()

    def eval_epoch(self, epoch, val_loader=None, calc_eval_std=False):
        with torch.no_grad():
            losses = []
            self.regressor.eval()
            self.deform_net.eval()
            self.vit.eval()
            self.resnet50.eval()

            if self.opt.log_eval_predictions is True:
                self.eval_predictions_data = pd.DataFrame(columns=[
                    "id", "reference_image_path", "distortion_image_path", "distortion_name", "label", "predicted_label"])

            # average the stats over multiple intrations
            iterations = 1 if self.opt.num_crop < 1 else self.opt.num_crop
            pred_epoch = np.zeros((len(val_loader), iterations, self.opt.val_batch_size))
            labels_epoch = np.zeros((len(val_loader), iterations, self.opt.val_batch_size))

            for iteration in range(iterations):
                for batch_index, data in enumerate(tqdm(val_loader)):
                    pred, labels = self.forward(data)
                    # save results in one epoch
                    pred_batch_numpy = pred.data.cpu().numpy().squeeze()
                    labels_batch_numpy = labels.data.cpu().numpy().squeeze()
                    pred_epoch[batch_index, iteration] = pred_batch_numpy
                    labels_epoch[batch_index, iteration] = labels_batch_numpy

            if self.opt.log_eval_predictions is True:
                for batch_index, data in enumerate(tqdm(val_loader)):
                    self.eval_predictions_data = pd.concat(
                        [self.eval_predictions_data,
                            pd.DataFrame({
                                "id": data.id.tolist(),
                                "reference_image_path": data.reference_image_path,
                                "distortion_image_path": data.distortion_image_path,
                                "distortion_name": data.distortion_name,
                            })], ignore_index=True)

            # compute loss over iterations
            pred_epoch_mean = np.mean(pred_epoch, axis=1)
            labels_epoch_mean = np.mean(labels_epoch, axis=1)
            pred_epoch_mean = np.reshape(pred_epoch_mean, (-1))
            labels_epoch_mean = np.reshape(labels_epoch_mean, (-1))
            loss = self.criterion(torch.tensor(pred_epoch_mean), torch.tensor(labels_epoch_mean))
            losses.append(loss.item())

            if self.opt.log_eval_predictions is True:
                data_len = len(self.eval_predictions_data)
                self.eval_predictions_data["label"] = labels_epoch_mean[:data_len]
                self.eval_predictions_data["predicted_label"] = pred_epoch_mean[:data_len]

            # compute correlation coefficient over iterations
            rho_s, _ = spearmanr(np.squeeze(pred_epoch_mean), np.squeeze(labels_epoch_mean))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch_mean), np.squeeze(labels_epoch_mean))
            print('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(
                epoch + 1, np.mean(losses), rho_s, rho_p))
            logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(
                epoch + 1, np.mean(losses), rho_s, rho_p))

            if calc_eval_std is True and iterations > 1:
                rho_s_std = self.get_metric_std(pred_epoch, labels_epoch, spearmanr, iterations)
                rho_p_std = self.get_metric_std(pred_epoch, labels_epoch, pearsonr, iterations)
                print('SRCC std:', rho_s_std)
                print('PLCC std:', rho_p_std)
                logging.info('SRCC std: {:.4}'.format(rho_s_std))
                logging.info('PLCC std: {:.4}'.format(rho_p_std))
            return np.mean(losses), rho_s, rho_p

    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'regressor_model_state_dict': self.regressor.state_dict(),
            'deform_net_model_state_dict': self.deform_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))

    def get_total_model_params(self):
        non_trainable_params_vit = sum(p.numel() for p in self.vit.eval().parameters())
        non_trainable_params_resnet = sum(p.numel() for p in self.resnet50.eval().parameters())
        trainable_params_deform = sum(p.numel() for p in self.deform_net.parameters() if p.requires_grad)
        trainable_params_regressor = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
        total_non_trainable_params = non_trainable_params_vit + non_trainable_params_resnet
        total_trainable_params = trainable_params_deform + trainable_params_regressor
        total_params = total_non_trainable_params + total_trainable_params

        print('Total non-trainable params:', total_non_trainable_params)
        print('Total trainable params:', total_trainable_params)
        print('Total params:', total_params)

    def get_total_params_size(self):
        # Calculate the total size of the model in MB
        models = [self.vit, self.resnet50, self.deform_net, self.regressor]
        params_size = 0
        for model in models:
            for param in model.parameters():
                params_size += param.element_size() * param.nelement()
        total_size = params_size * 1e-6
        print('Total model size:', total_size, 'MB')

    def load_model_from_path(self, models_dir):
        print(f'Loading model from {models_dir}...')
        if self.opt.evaluation_type == "traditional_datasets":
            checkpoint_path = os.path.join(models_dir, "best.pth")
        else:
            checkpoint_path = os.path.join(models_dir, "best.pth")
        checkpoint = torch.load(checkpoint_path)
        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
        self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def test(self, val_loader):
        print('Starting testing...')
        self.eval_epoch(0, val_loader, calc_eval_std=True)
        # self.eval_epoch(0, val_loader, calc_eval_std=True)
        dataset = val_loader.dataset
        dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) is True else dataset
        self.log_predictions(dataset)

    def get_metric_std(self, y_hat, y, metric_func, eval_iterations):
        results = [
            metric_func(y_hat[:, i, :].reshape(-1), y[:, i, :].reshape(-1))[0].item() for i in range(eval_iterations)]
        return torch.std(torch.tensor(results)).item()

    def log_predictions(self, dataset):
        split = dataset.split

        if self.eval_predictions_data is None:
            return

        file_path = os.path.join(self.opt.checkpoints_dir, f"{str(dataset)}_{split}_predictions.csv")
        self.eval_predictions_data.to_csv(file_path, index=True)
        self.eval_predictions_data = None
        print(f"Predictions logged to {file_path}")


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method("forkserver", force=True)
    config = TrainOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    logging.info(config)
    Train(config)
