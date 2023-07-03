import os
import argparse
import torch
import dill
# import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
# import pickle
# from environment.TrajectoryDS import TrajectoryDataset

from torch.utils.data import DataLoader
from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
# from models.trajectron import Trajectron
# from utils.model_registrar import ModelRegistrar
from utils import data_loader as LOADER
from utils.trajectron_hypers import get_traj_hypers
from utils.utils import mask_mse_func, post_process
from utils.visualization import plot_trajectory
import evaluation
import data.data as DATA
import data.dataset as DATASET
import torch.nn.functional as F
import utils.metrics as METRIC

class SPDiff():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True # explanation: https://blog.csdn.net/leviopku/article/details/121661020
        self._build()

    def train(self):
        
        for mode in range(1, int(self.config.finetune)+1):
            if mode == 0:
                train_loaders = self.train_loaders
                val_list = self.valid_list
                optimizer = self.optimizer
                scheduler = self.scheduler
                total_epochs = self.config.epochs
                finetune_flag = False
            elif mode == 1:
                train_loaders = self.finetune_train_loaders
                val_list = self.finetune_valid_list
                optimizer = self.ft_optimizer
                scheduler = self.ft_scheduler
                total_epochs = self.config.finetune_epochs
                finetune_flag = True
            train_data_num = int(len(train_loaders)*self.config.train_data_ratio)
            train_loaders = train_loaders[:train_data_num]
            loss_epoch = {}
            for epoch in range(total_epochs+1):
                if epoch!=0:
                    loss_epoch[epoch]=[]
                    # self.train_dataset.augment = self.config.augment
                    # if self.config.train_mode=='origin':
                    #     mse_list = []
                    #     # for node_type, data_loader in self.train_data_loader.items():
                    #     pbar = tqdm(list(itertools.islice(self.dataloader_train,10)), ncols=80)
                    #     node_type='pedestrian'
                    #     loader_len = len(pbar)
                    #     is_fst_loss = True
                    #     loss_batch = 0
                    #     batch_cnt = 0
                        
                    #     self.optimizer.zero_grad()
                    #     for cnt, batch in enumerate(pbar):
                            
                    #         train_loss = self.model.get_loss(batch, node_type)
                            
                    #         if (cnt+1)%self.config.batch_size !=0 and cnt != loader_len-1 :
                    #             if is_fst_loss :
                    #                 loss = train_loss
                    #                 is_fst_loss = False
                    #             else:
                    #                 loss += train_loss

                    #         else:
                    #             batch_cnt += 1
                    #             loss = train_loss # xiugai
                    #             loss = loss/self.config.batch_size*10000
                    #             is_fst_loss = True
                    #             loss.backward()
                    #             self.optimizer.step()
                    #             self.optimizer.zero_grad()
                    #             loss_batch += loss.item()
                    #             loss_epoch[epoch].append(loss.item())
                                
                    #             mse_list.append(loss.item())
                    #         pbar.set_description(f"Epoch {epoch}, {batch_cnt}/{len(pbar)//self.config.batch_size+1} MSE: {loss_batch/(batch_cnt if batch_cnt else 1):.10f}")
                    if self.config.train_mode=='multi':
                        mse_list = []
                        for i, train_loader in enumerate(train_loaders):
                            if self.config.finetune_trainmode=='singlestep' or not finetune_flag:
                                pbar = tqdm(train_loader, ncols=90)
                                # pbar = tqdm(list(itertools.islice(train_loaders,1)), ncols=90)
                                for batch_idx, batch_data in enumerate(pbar):
                                    self.optimizer.zero_grad()
                                    self.batch_idx = batch_idx
                                    train_loss = self.model.get_loss(batch_data)
                                    if self.config.diffnet!='SpatialTransformer_dest_force':
                                        train_loss.backward()
                                        optimizer.step()
                                        scheduler.step()
                                        print('last lr:',scheduler.get_last_lr())
                                    loss_epoch[epoch].append(train_loss.item())
                                    # mse_list.append(train_loss.item())
                                    pbar.set_description(f"Epoch {epoch}, {batch_idx+1}/{len(pbar)},{i+1}/{len(self.train_loaders)} MSE: {np.mean(loss_epoch[epoch])}")
                            elif self.config.finetune_trainmode=='multistep':
                                self.optimizer.zero_grad()
                                # torch.autograd.set_detect_anomaly(True)
                                assert type(train_loader)==DATA.ChanneledTimeIndexedPedData
                                loss = self.model.test_multiple_rollouts_for_training_geo(train_loader)
                                # loss = self.model.test_multiple_rollouts_for_training(train_loader)

                                loss_epoch[epoch].append(loss.item())
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                                print('last lr:',scheduler.get_last_lr())
                                mse_mean = np.mean(loss_epoch[epoch])
                                print(f"Epoch {epoch} NO {i} MSE: {mse_mean} ")
                                del loss
                                # import gc
                                # gc.collect()
                                # torch.cuda.empty_cache()
                        
                    self.log_writer.add_scalar('train_MSE', np.mean(loss_epoch[epoch]), epoch)
                
                # self.train_dataset.augment = False
                if (epoch) % self.config.eval_every == 0:
                    self.model.eval()

                    node_type = "PEDESTRIAN"
                    eval_ade_batch_errors = []
                    eval_fde_batch_errors = []
                    if self.config.train_mode =='origin':
                        pbar = tqdm(self.dataloader_eval)
                        i = 0
                        for batch in pbar:
                            i+=1
                            traj_pred = self.model.generate2(batch, node_type, num_points=0,sample=20,bestof=True)
                            gt = batch[3].squeeze().permute(2,0,1)
                    elif self.config.train_mode == 'multi':
                        if self.config.val_mode=='multistep':
                            mse_list = []
                            mae_list = []
                            ot_list = []
                            FDE_list = []
                            mmd_list = []
                            collision_list = []
                            dtw_list = []
                            ipd_list = []
                            ipd_mmd_list = []
                            for i, val_data in enumerate(val_list):
                                with torch.no_grad():
                                    traj_pred, dest_force, ped_force = self.model.generate_multistep_geo(val_data, t_start=self.config.skip_frames)
                                    # traj_pred, dest_force, ped_force = self.model.generate_multistep(val_data, t_start=self.config.skip_frames)
                                    
                                    ipd_mmd = METRIC.get_nearby_distance_mmd(traj_pred.position, traj_pred.velocity, 
                                                                   val_data.labels[..., :2], val_data.labels[..., 2:4], 
                                                                   val_data.mask_p_pred.long(), self.config.dist_threshold_ped, self.config.topk_ped*2, reduction='mean')
                                    ipd_mmd_list.append(ipd_mmd)
                                    p_pred = traj_pred.position
                                    # p_pred_ = p_pred.clone()
                                    # p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                                    # p_pred = p_pred_
                                    mask_p_pred = val_data.mask_p_pred.long()  # (*c) t, n
                                    labels = val_data.labels[..., :2]
                                    import time # debug
                                    
                                    torch.save(labels, self.logs_dir+f'/{epoch}_labels.pth')
                                    torch.save(p_pred, self.logs_dir+f'/{epoch}_p_pred.pth')
                                    torch.save(mask_p_pred, self.logs_dir+f'/{epoch}_mask_p_pred.pth')
                                    torch.save(dest_force, self.logs_dir+f'/{epoch}_dest_force.pth')
                                    torch.save(ped_force, self.logs_dir+f'/{epoch}_ped_force.pth')
                                    
                                    # plot_trajectory(p_pred, labels, name=time.strftime('%Y-%m-%d-%H-%M'))
                                    collision = METRIC.collision_count(p_pred, 0.5, reduction='sum')
                                    FDE = METRIC.fde_at_label_end(p_pred, labels, reduction='mean')
                                    
                                    p_pred = post_process(val_data, p_pred, traj_pred.mask_p, mask_p_pred)
                                    # pdb.set_trace()
                                    # p_pred[p_pred.isnan()]=0 
                                    a1 = time.time()
                                    dtw = METRIC.dtw_tensor(p_pred, labels, mask_p_pred, mask_p_pred,reduction='mean')
                                    dtw_list.append(dtw)
                                    a1 -= time.time()
                                    print(-a1)
                                    
                                    a2 = time.time()
                                    func = lambda x: x*torch.exp(-x)
                                    ipd = METRIC.inter_ped_dis(p_pred, labels, mask_p_pred,reduction='mean',applied_func=func)
                                    ipd_list.append(ipd)
                                    a2 -= time.time()
                                    print(-a2)
                                    
                                    loss = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1], reduction='mean')*2
                                    loss = loss.item()
                                    mse_list.append(loss)
                                    mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                                    mae_list.append(mae)
                                torch.save(p_pred,'post_pred.pt')
                                torch.save(labels,'post_labels.pt')
                                torch.save(mask_p_pred,'post_mask_p_pred.pt')

                                ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean', dvs=self.config.device)
                                mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                                
                                ot_list.append(ot)
                                mmd_list.append(mmd)
                                FDE_list.append(FDE)
                                collision_list.append(collision)
                                # ade,fde = evaluation.compute_batch_statistics2(traj_pred,gt,best_of=True)
                                # eval_ade_batch_errors.append(ade)
                                # eval_fde_batch_errors.append(fde)
                            # ade = np.mean(eval_ade_batch_errors)
                            # fde = np.mean(eval_fde_batch_errors) 
                            mse = np.mean(mse_list)
                            mae = np.mean(mae_list)
                            ot = np.mean(ot_list)
                            FDE = np.mean(FDE_list)
                            mmd = np.mean(mmd_list)
                            collision = np.mean(collision)
                            dtw = np.mean(dtw_list)
                            ipd = np.mean(ipd_list)
                            ipd_mmd = np.mean(ipd_mmd_list)
                            # if self.config.dataset == "eth":
                            #     ade = ade/0.6
                            #     fde = fde/0.6
                            # elif self.config.dataset == "sdd":
                            #     ade = ade * 50
                            #     fde = fde * 50
                            # print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
                            print(f"Epoch {epoch} MSE: {mse} MAE: {mae}")
                            print(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                            print(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                            print(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")
                            
                            # self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")
                            # self.log_writer.add_scalar('ADE', ade, epoch)
                            # self.log_writer.add_scalar('FDE', fde, epoch)
                            self.log.info(f"Epoch {epoch} MSE: {mse} MAE: {mae}")
                            self.log.info(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                            self.log.info(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                            self.log.info(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")
                            self.log.info(" ")
                            self.log_writer.add_scalar('MSE', mse, epoch)
                            self.log_writer.add_scalar('MAE', mae, epoch)
                            self.log_writer.add_scalar('OT', ot, epoch)
                            self.log_writer.add_scalar('MMD', mmd, epoch)
                            self.log_writer.add_scalar('Collision', collision, epoch)
                            self.log_writer.add_scalar('fde', FDE, epoch)
                            self.log_writer.add_scalar('dtw', dtw, epoch)
                            self.log_writer.add_scalar('ipd_mmd', ipd_mmd, epoch)
                            if self.config.save_model:
                                save_dir = os.path.join(self.logs_dir,'chpt')
                                print(f'save at epoch {epoch} to {save_dir}...')
                                os.makedirs(save_dir, exist_ok=True)
                                save_name = f'Epoch {epoch}' +'.pkl'
                                save_path = os.path.join(save_dir, save_name)
                                net_state_dict = self.model.state_dict()
                                torch.save(net_state_dict, save_path)
                            del traj_pred
                            # import gc
                            # gc.collect()
                            # torch.cuda.empty_cache()
                        elif self.config.val_mode=='singlestep':
                            if self.config.esti_goal == 'acce':
                                val_mses=[]
                                for i, val_data in enumerate(val_list):
                                    gt = val_data.labels[1:,:,4:6]
                                    hist_fea = val_data.self_hist_features[:-1,:,:,:6]
                                    ped_features = val_data.ped_features[:-1]
                                    obs_features = val_data.obs_features[:-1]
                                    self_feature = val_data.self_features[:-1,:,:]
                                    context = (hist_fea, ped_features, self_feature, obs_features)
                                    curr = val_data.labels[:-1,:,:6]
                                    traj_pred = self.model.generate_onestep(context,curr)
                                    mask = val_data.mask_a_pred[:-1]
                                    # mask[0] = mask[1]
                                    val_mse = mask_mse_func(traj_pred, gt, mask).item()
                                    print(f"Val Dataset {i} Epoch {epoch} val mse {val_mse}")
                                    self.log.info(f"Val Dataset {i} Epoch {epoch} val mse: {val_mse}")
                                    self.log_writer.add_scalar(f'val_mse_{i}', val_mse, epoch)
                                    val_mses.append(val_mse)
                            elif self.config.esti_goal == 'pos': 
                                raise NotImplementedError
                                for i, val_data in enumerate(self.valid_list):
                                    gt = val_data.labels[1:,:,:2]
                                    hist_pos = val_data.self_hist_features[:-1,:,:,:2]
                                    hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
                                    hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
                                    hist_vel[:,:,1:,:] = val_data.self_hist_features[:-1, :, :-1, 2:4]
                                    hist_acce[:,:,2:,:] = val_data.self_hist_features[:-1, :, :-2, 4:6]
                                    history = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
                                    dest_pos = val_data.self_features[:-1,:,:2]
                                    if self.config.history_dim==6:
                                        traj_pred = self.model.generate_onestep(history, dest_pos)
                                    elif self.config.history_dim==2:
                                        traj_pred = self.model.generate_onestep(history[...,:2], dest_pos)
                                    mask = dest_pos.abs().sum(dim=-1)==0 # t, N #TODO:**mas**dest_x=dest_y=******mas****）
                                    # mask[0] = mask[1]
                                    val_mse = mask_mse_func(traj_pred, gt, ~mask)
                                    print(f"Val Dataset {i} Epoch {epoch} val mse {val_mse}")
                                    self.log.info(f"Val Dataset {i} Epoch {epoch} val mse: {val_mse}")
                                    self.log_writer.add_scalar(f'val_mse_{i}', val_mse, epoch)
                            val_mse_mean = np.mean(val_mses)
                            if self.config.save_model:
                                save_dir = os.path.join(self.logs_dir,'ckpt')
                                print(f'save at epoch {epoch} to {save_dir}...')
                                os.makedirs(save_dir, exist_ok=True)
                                save_name = f'epoch{epoch}_val_mse_'+format(val_mse_mean,'.5f')+'.pkl'
                                save_path = os.path.join(save_dir, save_name)
                                net_state_dict = self.model.state_dict()
                                torch.save(net_state_dict, save_path)
                    
                    self.model.train()


    def eval(self):
        epoch = self.config.eval_at

        for _ in range(5):

            node_type = "PEDESTRIAN"
            eval_ade_batch_errors = []
            eval_fde_batch_errors = []
            ph = self.hyperparams['prediction_horizon']
            max_hl = self.hyperparams['maximum_history_length']


            for i, scene in enumerate(self.eval_scenes):
                print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t,t+10)
                    batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                   pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                   min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                                   max_ft=12, hyperparams=self.hyperparams)
                    if batch is None:
                        continue
                    test_batch = batch[0]
                    nodes = batch[1]
                    timesteps_o = batch[2]
                    traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True) # B * 20 * 12 * 2

                    predictions = traj_pred
                    predictions_dict = {}
                    for i, ts in enumerate(timesteps_o):
                        if ts not in predictions_dict.keys():
                            predictions_dict[ts] = dict()
                        predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))



                    batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=self.eval_env.NodeType,
                                                                           kde=False,
                                                                           map=None,
                                                                           best_of=True,
                                                                           prune_ph_to_future=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



            ade = np.mean(eval_ade_batch_errors)
            fde = np.mean(eval_fde_batch_errors)

            if self.config.dataset == "eth":
                ade = ade/0.6
                fde = fde/0.6
            elif self.config.dataset == "sdd":
                ade = ade * 50
                fde = fde * 50

            print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")


    def _build(self):
        self._build_dir()

        # self._build_encoder_config()
        # self._build_encoder()
        self._build_model()
        # self._build_train_loader()
        # self._build_eval_loader()
        if self.config.train_mode=='origin':
            self._build_train_loader2()
            self._build_eval_loader2()
        elif self.config.train_mode == 'multi':
            self._build_data_loader()
        else:
            raise NotImplementedError

        self._build_optimizer()
        

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        import sys
        debug_flag = 'run' if sys.gettrace() ==None else 'debug'
        print('running in',debug_flag,'mode')
        logs_dir = osp.join(self.model_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
        logs_dir += debug_flag
        self.logs_dir = logs_dir
        os.makedirs(logs_dir,exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=logs_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(logs_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)
        self.log.info(time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.log.info("Config:")
        for item in self.config.items():
            self.log.info(item)

        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([
                                    # {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                    {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.999)
        if 'ft_lr' not in self.config.keys():
            if 'ucy' in self.config.finetune_dict_path:
                self.config.ft_lr=self.config.lr/100
            elif 'gc' in self.config.finetune_dict_path:
                self.config.ft_lr=self.config.lr/1000
            else:
                raise NotImplementedError
            # self.config.ft_lr=self.config.lr/1000
            # self.config.ft_lr=self.config.lr
        print('ft_lr:',self.config.ft_lr)
        self.ft_optimizer = optim.Adam([
                                    # {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                    {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.ft_lr,
                                    weight_decay=1e-5)
        if 'ucy' in self.config.finetune_dict_path:
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.999)
        elif 'gc' in self.config.finetune_dict_path:
            # self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=10, gamma=0.999)
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.99)
        else:
            raise NotImplementedError
        self.log.info(f'(\'ft_lr\', {self.config.ft_lr})')
        print("> Optimizer built!")


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = None)
        self.model = model.cuda()
        # if self.config.eval_mode:
        #     self.model.load_state_dict(self.checkpoint['ddpm'])
        if self.config.model_sd_path:
            self.model.load_state_dict(torch.load(self.config.model_sd_path))
        train_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        trainable_params = np.sum([p.numel() for p in train_params])
        print('#Trainable Parameters:', trainable_params)
        self.log.info(f'#Trainable Parameters: {trainable_params}')
        self.log.info("\n")
        print("> Model built!")

   
    # def _build_train_loader2(self):
    #     with open(self.train_data_path, 'rb') as f:
    #         dset_train = dill.load(f)
         
    #     # dset_train = TrajectoryDataset(
    #     #     data_dir = './raw_data/'+self.config.dataset+'/train/',
    #     #     obs_len=self.config.obs_seq_len,
    #     #     pred_len=self.config.pred_seq_len,
    #     #     skip=1,norm_lap_matr=False,normalize=True)
    #     self.dataloader_train = DataLoader(
    #         dset_train,
    #         batch_size=1, #This is irrelative to the args batch size parameter
    #         shuffle =False,
    #         num_workers=0)
    #     return
    # def _build_eval_loader2(self):
    #     with open(self.eval_data_path, 'rb') as f:
    #         dset_eval = dill.load(f)
    #     # dset_eval = TrajectoryDataset(
    #     #     data_dir = './raw_data/'+self.config.dataset+'/eval/',
    #     #     obs_len=self.config.obs_seq_len,
    #     #     pred_len=self.config.pred_seq_len,
    #     #     skip=10,norm_lap_matr=False,normalize=True)
    #     self.dataloader_eval = DataLoader(
    #         dset_eval,
    #         batch_size=1, #This is irrelative to the args batch size parameter
    #         shuffle =False,
    #         num_workers=1)
    #     return
    def _build_data_loader(self):
        # if self.config.rebuild_dataset == True:
        #     if self.config.dataset_type=='timeindex':
        #         synthetic_dataset = DATASET.TimeIndexedPedDataset2()    
        #     else:
        #         raise NotImplementedError
        #     synthetic_dataset.load_data(self.config.data_path)
            
        #     print('number of training dataset: ', len(synthetic_dataset.raw_data['train']))
        #     synthetic_dataset.build_dataset(self.config, finetune_flag=False)
            
        #     with open(self.config.data_dict_path, 'wb') as f:
        #         dill.dump(synthetic_dataset, f, protocol=dill.HIGHEST_PROTOCOL)
        # else:
        #     with open(self.config.data_dict_path, 'rb') as f:
        #         synthetic_dataset = dill.load(f)
        
        if self.config.rebuild_finetune_dataset == True:
            if self.config.dataset_type=='timeindex':
                finetune_dataset = DATASET.TimeIndexedPedDataset2()    
            else:
                raise NotImplementedError
            finetune_dataset.load_data(self.config.finetune_data_path)
            
            print('number of finetune training dataset: ', len(finetune_dataset.raw_data['train']))
            finetune_dataset.build_dataset(self.config, finetune_flag=(self.config.finetune_trainmode=='multistep'))
            
            with open(self.config.finetune_dict_path, 'wb') as f:
                dill.dump(finetune_dataset, f, protocol=dill.HIGHEST_PROTOCOL)
        elif self.config.finetune == True:
            with open(self.config.finetune_dict_path, 'rb') as f:
                finetune_dataset = dill.load(f)
        
        # self.train_loaders=[]
        
        # train_list = synthetic_dataset.train_data
        # if type(train_list)==list:
        #     for item in train_list:
        #         self.train_loaders.append(DataLoader(
        #                                     item,
        #                                     batch_size=self.config.batch_size, 
        #                                     shuffle =False,
        #                                     drop_last=True))

        # else:
        #     raise NotImplementedError
        

        # self.valid_list = synthetic_dataset.valid_data

        
        if self.config.finetune == True:
            
            finetune_train_list = finetune_dataset.train_data
            if self.config.finetune_trainmode=='singlestep':
                self.finetune_train_loaders=[]
                assert type(finetune_train_list)==list
                for item in finetune_train_list:
                    self.finetune_train_loaders.append(DataLoader(
                                                item,
                                                batch_size=self.config.batch_size, 
                                                shuffle =False,
                                                drop_last=True))
            elif self.config.finetune_trainmode=='multistep':
                assert type(finetune_train_list[0])==DATA.ChanneledTimeIndexedPedData
                self.finetune_train_loaders = LOADER.data_loader(finetune_train_list, self.config.batch_size,
                                                self.config.seed, shuffle=False, drop_last=False)

            
            # self.finetune_valid_loaders=[]
            self.finetune_valid_list = finetune_dataset.valid_data
            # if type(self.finetune_valid_list)==list:
            #     for item in self.finetune_valid_list:
            #         self.finetune_valid_loaders.append(DataLoader(
            #                             item,
            #                             batch_size=self.config.batch_size, #This is irrelative to the args batch size parameter
            #                             shuffle =False,
            #                             drop_last=True))
            # elif type(self.finetune_valid_list)==DATA.PointwisePedData:
            #     self.finetune_valid_loaders = self.finetune_valid_list
        return
        
        
    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")
