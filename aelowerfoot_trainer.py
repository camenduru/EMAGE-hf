import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation


class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.joints = self.train_data.joints
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        self.tracker = other_tools.EpochTracker(["rec", "contact", "vel", "foot", "ver", "com", "kl", "acc", "trans", "transv"], [False,False, False, False, False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
    
    def inverse_selection(self, filtered_t, selection_array, n):
        # 创建一个全为零的数组，形状为 n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # 找到选择数组中为1的索引位置
        selected_indices = np.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # 创建一个全为零的数组，形状为 n*165
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        
        # 找到选择数组中为1的索引位置
        selected_indices = torch.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t


    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose_raw = dict_data["pose"]
            tar_beta = dict_data["beta"].cuda()
            tar_trans = dict_data["trans"].cuda()
            # tar_trans_vel_x = other_tools.estimate_linear_velocity(tar_trans[:, :, 0:1], dt=1/self.args.pose_fps)
            # tar_trans_vel_z = other_tools.estimate_linear_velocity(tar_trans[:, :, 2:3], dt=1/self.args.pose_fps)
            tar_pose = tar_pose_raw[:, :, :27].cuda() 
            tar_contact = tar_pose_raw[:, :, 27:31].cuda() 
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            tar_exps = torch.zeros((bs, n, 100)).cuda()
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
            tar_trans_copy = tar_trans
            tar_contact_copy = tar_contact
            in_tar_pose = torch.cat((tar_pose, tar_trans_copy, tar_contact_copy), dim=-1)
            
            t_data = time.time() - t_start 
            
            self.opt.zero_grad()
            g_loss_final = 0
            net_out = self.model(in_tar_pose)
            rec_pose = net_out["rec_pose"][:, :, :j*6]
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
            loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss_final += loss_rec

            rec_contact = net_out["rec_pose"][:, :, j*6+3:j*6+7]
            loss_contact = self.vectices_loss(rec_contact, tar_contact) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("contact", "train", loss_contact.item())
            g_loss_final += loss_contact 

            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight
            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            g_loss_final += velocity_loss 
            g_loss_final += acceleration_loss 

            # rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
            # rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            # rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            # rec_y_trans = rec_trans[:,:,1:2]
            # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            # loss_trans_vel = self.vel_loss(rec_trans[:, :, 0:1], tar_trans_vel_x) * self.args.rec_weight \
            # + self.vel_loss(rec_trans[:, :, 2:3], tar_trans_vel_z) * self.args.rec_weight 
            # v3 =  self.vel_loss(rec_trans[:, :, 0:1][:, 1:] - rec_trans[:, :, 0:1][:, :-1], tar_trans_vel_x[:, 1:] - tar_trans_vel_x[:, :-1]) * self.args.rec_weight \
            # + self.vel_loss(rec_trans[:, :, 2:3][:, 1:] - rec_trans[:, :, 2:3][:, :-1], tar_trans_vel_z[:, 1:] - tar_trans_vel_z[:, :-1]) * self.args.rec_weight
            # a3 = self.vel_loss(rec_trans[:, :, 0:1][:, 2:] + rec_trans[:, :, 0:1][:, :-2] - 2 * rec_trans[:, :, 0:1][:, 1:-1], tar_trans_vel_x[:, 2:] + tar_trans_vel_x[:, :-2] - 2 * tar_trans_vel_x[:, 1:-1]) * self.args.rec_weight \
            # + self.vel_loss(rec_trans[:, :, 2:3][:, 2:] + rec_trans[:, :, 2:3][:, :-2] - 2 * rec_trans[:, :, 2:3][:, 1:-1], tar_trans_vel_z[:, 2:] + tar_trans_vel_z[:, :-2] - 2 * tar_trans_vel_z[:, 1:-1]) * self.args.rec_weight
            # g_loss_final += 5*v3 
            # g_loss_final += 5*a3
            # v2 =  self.vel_loss(rec_xyz_trans[:, 1:] - rec_xyz_trans[:, :-1], tar_trans[:, 1:] - tar_trans[:, :-1]) * self.args.rec_weight
            # a2 =  self.vel_loss(rec_xyz_trans[:, 2:] + rec_xyz_trans[:, :-2] - 2 * rec_xyz_trans[:, 1:-1], tar_trans[:, 2:] + tar_trans[:, :-2] - 2 * tar_trans[:, 1:-1]) * self.args.rec_weight
            # g_loss_final += 5*v2 
            # g_loss_final += 5*a2 
            # self.tracker.update_meter("transv", "train", loss_trans_vel.item())
            # g_loss_final += loss_trans_vel
            # loss_trans = self.vel_loss(rec_xyz_trans, tar_trans) * self.args.rec_weight 
            # self.tracker.update_meter("trans", "train", loss_trans.item())
            # g_loss_final += loss_trans

             # vertices loss
            if self.args.rec_ver_weight > 0:
                # print(tar_pose.shape, bs, n, j)
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_verts=False,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices_tar = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=False,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )  
                joints_rec = vertices_rec['joints']
                # print(joints_rec.shape)
                joints_rec = joints_rec.reshape(bs, n, -1, 3)
                vectices_loss = self.vectices_loss(vertices_rec['joints'], vertices_tar['joints'])
                foot_idx = [7, 8, 10, 11]
                model_contact = net_out["rec_pose"][:, :, j*6+3:j*6+7]
                # find static indices consistent with model's own predictions
                static_idx = model_contact > 0.95  # N x S x 4
                # print(model_contact,static_idx)
                model_feet = joints_rec[:, :, foot_idx]  # foot positions (N, S, 4, 3)
                model_foot_v = torch.zeros_like(model_feet)
                model_foot_v[:, :-1] = (
                    model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
                )  # (N, S-1, 4, 3)
                model_foot_v[~static_idx] = 0
                foot_loss = self.vel_loss(
                    model_foot_v, torch.zeros_like(model_foot_v)
                )
                self.tracker.update_meter("foot", "train", foot_loss.item()*self.args.rec_weight * self.args.rec_ver_weight*20)
                self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                g_loss_final += (vectices_loss)*self.args.rec_weight*self.args.rec_ver_weight 
                g_loss_final += foot_loss*self.args.rec_weight*self.args.rec_ver_weight*20 
            
            # ---------------------- vae -------------------------- #
            if "VQVAE" in self.args.g_name:
                loss_embedding = net_out["embedding_loss"]
                g_loss_final += loss_embedding
                self.tracker.update_meter("com", "train", loss_embedding.item())
            # elif "VAE" in self.args.g_name:
            #     pose_mu, pose_logvar = net_out["pose_mu"], net_out["pose_logvar"] 
            #     KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
            #     if epoch < 0:
            #         KLD_weight = 0
            #     else:
            #         KLD_weight = min(1.0, (epoch - 0) * 0.05) * 0.01
            #     loss += KLD_weight * KLD
            #     self.tracker.update_meter("kl", "train", KLD_weight * KLD.item())    
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                tar_pose_raw = dict_data["pose"]
                tar_beta = dict_data["beta"].cuda()
                tar_trans = dict_data["trans"].cuda()
                tar_trans_vel_x = other_tools.estimate_linear_velocity(tar_trans[:, :, 0:1], dt=1/self.args.pose_fps)
                tar_trans_vel_z = other_tools.estimate_linear_velocity(tar_trans[:, :, 2:3], dt=1/self.args.pose_fps)
                #print(tar_pose.shape)
                tar_pose = tar_pose_raw[:, :, :27].cuda() 

                tar_contact = tar_pose_raw[:, :, 27:31].cuda()  
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_exps = torch.zeros((bs, n, 100)).cuda()
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                tar_trans_copy = tar_trans
                tar_contact_copy = tar_contact
                in_tar_pose = torch.cat((tar_pose, tar_trans_copy, tar_contact_copy), dim=-1)
                t_data = time.time() - t_start 

                #self.opt.zero_grad()
                #g_loss_final = 0
                net_out = self.model(in_tar_pose)
                rec_pose = net_out["rec_pose"][:, :, :j*6]
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
                rec_contact = net_out["rec_pose"][:, :, j*6+3:j*6+7]
                # print(rec_contact.shape, tar_contact.shape)
                loss_contact = self.vel_loss(rec_contact, tar_contact) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("contact", "val", loss_contact.item())
                #g_loss_final += loss_rec
                rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
                rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
                rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
                rec_y_trans = rec_trans[:,:,1:2]
                rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)

                # rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
                # rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
                # rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
                # rec_y_trans = rec_trans[:,:,1:2]
                # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
                # loss_trans_vel = self.vel_loss(rec_trans[:, :, 0:1], tar_trans_vel_x) * self.args.rec_weight \
                # + self.vel_loss(rec_trans[:, :, 2:3], tar_trans_vel_z) * self.args.rec_weight 
                # v3 =  self.vel_loss(rec_trans[:, :, 0:1][:, 1:] - rec_trans[:, :, 0:1][:, :-1], tar_trans_vel_x[:, 1:] - tar_trans_vel_x[:, :-1]) * self.args.rec_weight \
                # + self.vel_loss(rec_trans[:, :, 2:3][:, 1:] - rec_trans[:, :, 2:3][:, :-1], tar_trans_vel_z[:, 1:] - tar_trans_vel_z[:, :-1]) * self.args.rec_weight
                # a3 = self.vel_loss(rec_trans[:, :, 0:1][:, 2:] + rec_trans[:, :, 0:1][:, :-2] - 2 * rec_trans[:, :, 0:1][:, 1:-1], tar_trans_vel_x[:, 2:] + tar_trans_vel_x[:, :-2] - 2 * tar_trans_vel_x[:, 1:-1]) * self.args.rec_weight \
                # + self.vel_loss(rec_trans[:, :, 2:3][:, 2:] + rec_trans[:, :, 2:3][:, :-2] - 2 * rec_trans[:, :, 2:3][:, 1:-1], tar_trans_vel_z[:, 2:] + tar_trans_vel_z[:, :-2] - 2 * tar_trans_vel_z[:, 1:-1]) * self.args.rec_weight
                # #g_loss_final += 5*v3 
                # #g_loss_final += 5*a3
                # v2 =  self.vel_loss(rec_xyz_trans[:, 1:] - rec_xyz_trans[:, :-1], tar_trans[:, 1:] - tar_trans[:, :-1]) * self.args.rec_weight
                # a2 =  self.vel_loss(rec_xyz_trans[:, 2:] + rec_xyz_trans[:, :-2] - 2 * rec_xyz_trans[:, 1:-1], tar_trans[:, 2:] + tar_trans[:, :-2] - 2 * tar_trans[:, 1:-1]) * self.args.rec_weight
                #g_loss_final += 5*v2 
                #g_loss_final += 5*a2 
                # self.tracker.update_meter("transv", "val", loss_trans_vel.item())
                # #g_loss_final += loss_trans_vel
                # loss_trans = self.vel_loss(rec_xyz_trans, tar_trans) * self.args.rec_weight 
                # self.tracker.update_meter("trans", "val", loss_trans.item())
                #g_loss_final += loss_trans

                 # vertices loss
                if self.args.rec_ver_weight > 0:
                    # print(tar_pose.shape, bs, n, j)
                    tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                    tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_verts=False, 
                        return_joints=True,
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=tar_pose[:, 66:69], 
                        global_orient=tar_pose[:,:3], 
                        body_pose=tar_pose[:,3:21*3+3], 
                        left_hand_pose=tar_pose[:,25*3:40*3], 
                        right_hand_pose=tar_pose[:,40*3:55*3], 
                        return_verts=False, 
                        return_joints=True,
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )  
                    joints_rec = vertices_rec['joints']
                    joints_rec = joints_rec.reshape(bs, n, -1, 3)
                    vectices_loss = self.vectices_loss(vertices_rec['joints'], vertices_tar['joints'])
                    foot_idx = [7, 8, 10, 11]
                    model_contact = net_out["rec_pose"][:, :, j*6+3:j*6+7]
                    # find static indices consistent with model's own predictions
                    static_idx = model_contact > 0.95  # N x S x 4
                    # print(model_contact)
                    model_feet = joints_rec[:, :, foot_idx]  # foot positions (N, S, 4, 3)
                    model_foot_v = torch.zeros_like(model_feet)
                    model_foot_v[:, :-1] = (
                        model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
                    )  # (N, S-1, 4, 3)
                    model_foot_v[~static_idx] = 0
                    foot_loss = self.vectices_loss(
                        model_foot_v, torch.zeros_like(model_foot_v)
                    )
                    self.tracker.update_meter("foot", "val", foot_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                    self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                if "VQVAE" in self.args.g_name:
                    loss_embedding = net_out["embedding_loss"]
                    self.tracker.update_meter("com", "val", loss_embedding.item())
                    #g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight
                if self.args.debug:
                    if its == 1: break
            self.val_recording(epoch)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in enumerate(self.test_loader):
                tar_pose_raw = dict_data["pose"]
                tar_trans = dict_data["trans"].to(self.rank)
                tar_pose = tar_pose_raw[:, :, :27].cuda() 
                tar_contact = tar_pose_raw[:, :, 27:31].cuda() 
                # tar_pose = tar_pose.cuda()
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]
                tar_contact = tar_contact[:, :n-remain, :]
                tar_trans_copy = tar_trans[:, :n-remain, :]
                tar_contact_copy = tar_contact
                in_tar_pose = torch.cat([tar_pose, tar_trans_copy, tar_contact_copy], dim=-1)
                #print(tar_pose.shape)
                if True:
                    net_out = self.model(in_tar_pose)
                    rec_pose = net_out["rec_pose"][:, :, :j*6]
                    rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3] - net_out["rec_pose"][:, :, j*6:j*6+3]
                    # print(rec_trans.shape)
                    rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
                    rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
                    rec_y_trans = rec_trans[:,:,1:2]
                    rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
                    n = rec_pose.shape[1]
                    rec_trans = rec_trans.cpu().numpy().reshape(bs*n, 3)
                    tar_pose = tar_pose[:, :n, :]
                    rec_pose = rec_pose.reshape(bs, n, j, 6) 
                    rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = rec_pose.cpu().numpy()
                else:
                    pass
#                     for i in range(tar_pose.shape[1]//(self.args.vae_test_len)):
#                         tar_pose_new = tar_pose[:,i*(self.args.vae_test_len):i*(self.args.vae_test_len)+self.args.vae_test_len,:]
#                         net_out = self.model(**dict(inputs=tar_pose_new))
#                         rec_pose = net_out["rec_pose"]
#                         rec_pose = (rec_pose.reshape(rec_pose.shape[0], rec_pose.shape[1], -1, 6) * self.joint_level_mask_cuda).reshape(rec_pose.shape[0], rec_pose.shape[1], -1)
#                         if "rot6d" in self.args.pose_rep:
#                             rec_pose = data_transfer.rotation_6d_to_matrix(rec_pose.reshape(tar_pose.shape[0], self.args.vae_test_len, -1, 6))
#                             rec_pose = data_transfer.matrix_to_euler_angles(rec_pose, "XYZ").reshape(rec_pose.shape[0], rec_pose.shape[1], -1)
#                             if "smplx" not in self.args.pose_rep:
#                                 rec_pose = torch.rad2deg(rec_pose)
#                             rec_pose = rec_pose * self.joint_mask_cuda
                            
#                         out_sub = rec_pose.cpu().numpy().reshape(-1, rec_pose.shape[2])
#                         if i != 0:
#                             out_final = np.concatenate((out_final,out_sub), 0)
#                         else:
#                             out_final = out_sub
                            
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                tar_pose = tar_pose.cpu().numpy()
                
                total_length += n 
                # --- save --- #
                if 'smplx' in self.args.pose_rep:
                    gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+'.npz', allow_pickle=True)
                    stride = int(30 / self.args.pose_fps)
                    tar_pose = self.inverse_selection(tar_pose, self.test_data.joint_mask, tar_pose.shape[0])
                    np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                        betas=gt_npz["betas"],
                        poses=tar_pose[:n],
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=rec_trans-rec_trans,
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )
                    rec_pose = self.inverse_selection(rec_pose, self.test_data.joint_mask, rec_pose.shape[0])
                    np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                        betas=gt_npz["betas"],
                        poses=rec_pose,
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=rec_trans-rec_trans,
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )       
                else:
                    rec_pose = rc.axis_angle_to_matrix(torch.from_numpy(rec_pose.reshape(bs*n, j, 3)))
                    rec_pose = np.rad2deg(rc.matrix_to_euler_angles(rec_pose, "XYZ")).reshape(bs*n, j*3).numpy()                
                    tar_pose = rc.axis_angle_to_matrix(torch.from_numpy(tar_pose.reshape(bs*n, j, 3)))
                    tar_pose = np.rad2deg(rc.matrix_to_euler_angles(tar_pose, "XYZ")).reshape(bs*n, j*3).numpy() 
                    #trans="0.000000 0.000000 0.000000"
                    
                    with open(f"{self.args.data_path}{self.args.pose_rep}/{test_seq_list.iloc[its]['id']}.bvh", "r") as f_demo:
                        with open(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.bvh', 'w+') as f_gt:
                            with open(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.bvh', 'w+') as f_real:
                                for i, line_data in enumerate(f_demo.readlines()):
                                    if i < 431:
                                        f_real.write(line_data)
                                        f_gt.write(line_data)
                                    else: break
                                for line_id in range(n): #,args.pre_frames, args.pose_length
                                    line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                    f_real.write(line_data[1:-2]+'\n')
                                for line_id in range(n): #,args.pre_frames, args.pose_length
                                    line_data = np.array2string(tar_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                                    f_gt.write(line_data[1:-2]+'\n')
                # with open(results_save_path+"gt_"+test_seq_list[its]+'.pkl', 'wb') as fw:
                #     pickle.dump(new_dict, fw)
                # #new_dict2["fullpose"] = out_final
                # with open(results_save_path+"res_"+test_seq_list[its]+'.pkl', 'wb') as fw1:
                #     pickle.dump(new_dict2, fw1)

                # other_tools.render_one_sequence(
                #     results_save_path+"res_"+test_seq_list[its]+'.pkl',
                #     results_save_path+"gt_"+test_seq_list[its]+'.pkl',
                #     results_save_path,
                #     self.args.data_path + self.args.test_data_path + 'wave16k/' + test_seq_list[its]+'.npy',
                # )
                                                                                                
                #if its == 1:break
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")