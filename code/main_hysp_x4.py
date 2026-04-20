import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import numpy as np
import random
import sys
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hyspecnet_dataset.hyspecnet11k import HySpecNet11k
from config_x4 import Config
from final_model_hysp_x4 import Model
from loss import HLoss
from metrics import compare_mpsnr


#-------------------------------------------------------Setup-------------------------------------------------------#
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '12356' 
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def create_datasets():
    config= Config()
    print('===> Loading datasets')

    train_dataset = HySpecNet11k(config.dataset, mode=config.mode, split="train", transform=None, augment=True)
    val_dataset = HySpecNet11k(config.dataset, mode=config.mode, split="val", transform=None)
    test_dataset = HySpecNet11k(config.dataset, mode=config.mode, split="test", transform=None)

    return train_dataset, val_dataset, test_dataset

    
def main(rank, world_size,train_dataset, val_dataset, test_dataset):
    
    setup(rank, world_size)     

    model_time = time.strftime("%Y%m%d_%H%M")

    config = Config()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    local_seed = config.seed + rank 
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)
    # Synchronize processes
    dist.barrier()

    torch.use_deterministic_algorithms(False)
        #-------------------------------------------------------Dataset-------------------------------------------------------#
    def worker_init_fn(worker_id):
        np.random.seed((local_seed))

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed = local_seed)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=config.batch_size, num_workers=config.workers, shuffle=False,
                                  pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn) 
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle = False, seed = local_seed)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=1, num_workers=config.workers, shuffle=False,
                                 pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle = False, seed = local_seed)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, num_workers=config.workers, shuffle=False,
                                  pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    #-------------------------------------------------------Loading checkpoint-------------------------------------------------------#
    def load_pretrained_model(model, optimizer, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded: resuming training from epoch {start_epoch}")
        return model, start_epoch
    #-------------------------------------------------------CreateModel-------------------------------------------------------#
    config.device = torch.device(f'cuda:{rank}')
    model = Model().to(config.device)
    if torch.cuda.is_available():
        model.cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    #-------------------------------------------------------Gpus-------------------------------------------------------#  
    if config.resume_training:
        checkpoint_path = config.checkpoint_path
        if not checkpoint_path:
            raise ValueError("`config.checkpoint_path` must be set when `resume_training` is True.")
        model, start_epoch = load_pretrained_model(model, optimizer, checkpoint_path, config.device)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        start_epoch = 0
    #-------------------------------------------------------N.TrainableParams-------------------------------------------------------#
    print('No. params: %d' % (sum(p.numel() for p in model.parameters() if p.requires_grad),) )
    #-------------------------------------------------------Logging-------------------------------------------------------#
    log_dir =os.path.join(config.log_dir,'DPSR_Hyspecnetx4_'+model_time)
    log_writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None
    #-------------------------------------------------------TraininingProcess-------------------------------------------------------#
    h_loss = HLoss(0.3,0.1)

    max_psnr=0.0
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = config.learning_rate
        
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        start_time = time.time()
        for step, x_hr in enumerate(tqdm(train_dataloader, leave=False)):            
            model.train()            
            accumulation_steps = config.acc_steps
            if (step) % accumulation_steps == 0:
                optimizer.zero_grad()            
            x_hr = x_hr.to(config.device)
           #------------------------------------------------DownsamplingByFactor4------------------------------------------------# 
            with torch.no_grad():            
                x_lr = F.interpolate(x_hr, scale_factor=0.25, mode='bicubic', align_corners=False, antialias=True)                
                x_lr = x_lr.to(config.device)                
            #------------------------------------------------ImageSuperRes------------------------------------------------#            
            x_sr = model(x_lr)    
            #------------------------------------------------Loss------------------------------------------------#            
            x_hr_senza_ultime_4_linee = rearrange(x_hr, "B C H W-> B C W H")            
            x_hr_senza_ultime_4_linee = x_hr_senza_ultime_4_linee[:, :, :, :-4]            
            x_hr_senza_ultime_4_linee = rearrange(x_hr_senza_ultime_4_linee, "B C W H-> B C H W")         
            loss= h_loss(x_sr, x_hr_senza_ultime_4_linee)/ accumulation_steps       
            loss.backward()            
            nn.utils.clip_grad_norm_(model.parameters(), 1)            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
        end_time = time.time()  
        epoch_time = end_time - start_time 
        print(f"Epoch [{epoch}] completed in {epoch_time / 60:.2f} minutes, train/loss: {loss.cpu().detach().numpy()}")

        if log_writer is not None:
            log_writer.add_scalar('train/loss', loss.cpu().detach().numpy(), epoch)

        if rank == 0 and epoch % config.save_every == 0 and epoch > 1 : 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_name = 'DPSR_Hyspecnetx4'
            checkpoint_path = f'{checkpoint_dir}/epoch_{epoch}_model_{model_name}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    #-------------------------------------------------------EvaluationProcess-------------------------------------------------------#        
        if epoch % config.validate_every == 0:
                model.eval()
                with torch.no_grad():                    
                    psnr_val=0                   
                    j_val= 0. 
                                       
                    for val_step, x_hr1_val in enumerate(tqdm(val_dataloader, leave=False)):
 
                        x_hr1_val = x_hr1_val.to(config.device)
                        x_lr_val = F.interpolate(x_hr1_val, scale_factor=1/4, mode='bicubic', align_corners=False, antialias=True)                        

                        x_sr_val = model(x_lr_val)
                        
                        x_sr_val = torch.clamp(x_sr_val, 0, 1)                       
                        x_hr1_val = rearrange(x_hr1_val, "B C H W-> B C W H")            
                        x_hr1_val = x_hr1_val[:, :, :, :-4]                        
                        x_hr1_val = rearrange(x_hr1_val, "B C W H-> B C H W")                        
                        for i_val in range(x_hr1_val.shape[0]):                            
                            single_image_sr_val = x_sr_val[i_val]                        
                            single_image_hr_val = x_hr1_val[i_val] 
                            
                            # ONLY FOR HYSPECNET-11K: Create a mask to identify non-zero channels in the high-resolution image
                            non_zero_mask = (single_image_hr_val.abs().sum(dim=(1, 2)) > 0)
                            # Use the mask to filter out zero channels
                            filtered_image_sr = single_image_sr_val[non_zero_mask]
                            filtered_image_hr = single_image_hr_val[non_zero_mask]
                            
                            filtered_image_hr = filtered_image_hr.cpu().numpy().transpose(1, 2, 0)
                            filtered_image_sr = filtered_image_sr.cpu().numpy().transpose(1, 2, 0)

                            psnr_per_channel_filtered =compare_mpsnr(filtered_image_hr, filtered_image_sr, data_range=1.)
                      
                            psnr_val += psnr_per_channel_filtered
                     
                            j_val += 1         
              
                    local_psnr_tensor_val = torch.tensor([psnr_val], device=config.device)

                    num_samples_tensor_val = torch.tensor([j_val], device=config.device)
                   
                    dist.all_reduce(local_psnr_tensor_val, op=dist.ReduceOp.SUM)

                    dist.all_reduce(num_samples_tensor_val, op=dist.ReduceOp.SUM)
            
                    global_psnr_val = local_psnr_tensor_val.item() / num_samples_tensor_val.item()
                    
                    if global_psnr_val > max_psnr:
                        max_psnr = global_psnr_val
                        if rank == 0:
                            checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            checkpoint_dir = 'checkpoints'
                            if not os.path.exists(checkpoint_dir):
                                os.makedirs(checkpoint_dir)
                            model_name = 'DPSR_Hyspecnetx4_best'
                            checkpoint_path = f'{checkpoint_dir}/epoch_{epoch}_model_{model_name}.pth'
                            torch.save(checkpoint, checkpoint_path)
                            print(f'New best model saved to {checkpoint_path} with PSNR: {global_psnr_val}')

                    if log_writer is not None:
                        log_writer.add_scalar('val PSNR medio su val set', global_psnr_val, epoch)
                        print(f"{epoch} -- PSNR val: {global_psnr_val} " ) 
        
        
        if epoch == config.epochs - 1:
                model.eval()
                with torch.no_grad():                    
                    psnr_test=0                   
                    j_test= 0. 
                                       
                    for test_step, x_hr1_test in enumerate(tqdm(test_dataloader, leave=False)):
 
                        x_hr1_test = x_hr1_test.to(config.device)
                        x_lr_test = F.interpolate(x_hr1_test, scale_factor=1/4, mode='bicubic', align_corners=False, antialias=True)                        

                        #   Line by line evaluation code
                        x_sr_test = []                        
                        for i in range(x_lr_test.shape[2]):                        
                            x_sr_i_test = model(x_lr_test[:,:,i:i+1,:], inference=True, index_initialize_states = i)                             
                            x_sr_test.append(x_sr_i_test)            
                        x_sr_test = torch.cat(x_sr_test, dim=2)
                        x_sr_test = x_sr_test[:, :, 4:, :]
                        
                        x_sr_test = torch.clamp(x_sr_test, 0, 1)                       
                        x_hr1_test = rearrange(x_hr1_test, "B C H W-> B C W H")            
                        x_hr1_test = x_hr1_test[:, :, :, :-4]                        
                        x_hr1_test = rearrange(x_hr1_test, "B C W H-> B C H W")                        
                        for i_test in range(x_hr1_test.shape[0]):                            
                            single_image_sr_test = x_sr_test[i_test]                        
                            single_image_hr_test = x_hr1_test[i_test] 
                            
                            # ONLY FOR HYSPECNET-11K: Create a mask to identify non-zero channels in the high-resolution image
                            non_zero_mask = (single_image_hr_test.abs().sum(dim=(1, 2)) > 0)
                            # Use the mask to filter out zero channels
                            filtered_image_sr = single_image_sr_test[non_zero_mask]
                            filtered_image_hr = single_image_hr_test[non_zero_mask]
                            
                            filtered_image_hr = filtered_image_hr.cpu().numpy().transpose(1, 2, 0)
                            filtered_image_sr = filtered_image_sr.cpu().numpy().transpose(1, 2, 0)

                            psnr_per_channel_filtered =compare_mpsnr(filtered_image_hr, filtered_image_sr, data_range=1.)
                      
                            psnr_test += psnr_per_channel_filtered
                     
                            j_test += 1         
              
                    local_psnr_tensor_test = torch.tensor([psnr_test], device=config.device)

                    num_samples_tensor_test = torch.tensor([j_test], device=config.device)
                   
                    dist.all_reduce(local_psnr_tensor_test, op=dist.ReduceOp.SUM)

                    dist.all_reduce(num_samples_tensor_test, op=dist.ReduceOp.SUM)
            
                    global_psnr_test = local_psnr_tensor_test.item() / num_samples_tensor_test.item()

                    if log_writer is not None:
                        log_writer.add_scalar('val PSNR medio su test set', global_psnr_test, epoch)
                        print(f"{epoch} -- PSNR test: {global_psnr_test} " ) 

    if log_writer is not None:
        log_writer.close()
    cleanup()
#------------------------------------------------------------MainInst------------------------------------------------------------#
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available. This script requires at least one GPU.")
    train_dataset, val_dataset, test_dataset = create_datasets() 
    torch.multiprocessing.spawn(main, args=(world_size,train_dataset, val_dataset, test_dataset), nprocs=world_size, join=True)
    
    
    
    
