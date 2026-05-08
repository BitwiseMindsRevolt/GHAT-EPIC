import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_epic as opts
import time
import h5py
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet, SuppressDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, regress_loss_func, suppress_loss_func
from tqdm import tqdm

def setup_multi_gpu():
    """Setup multi-GPU environment"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return num_gpus
    return 0

def train_one_epoch(opt, model, train_dataset, optimizer):
    # Increase num_workers for multi-GPU setup
    num_workers = min(8, os.cpu_count())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=num_workers, pin_memory=True,
                                                drop_last=False)      
    epoch_cost = 0
    
    for n_iter,(input_data,label) in enumerate(tqdm(train_loader)):
        # Move data to GPU (DataParallel will handle distribution)
        input_data = input_data.cuda()
        label = label.cuda()
        
        suppress_conf = model(input_data)
        
        loss = suppress_loss_func(label,suppress_conf)
        epoch_cost+= loss.detach().cpu().numpy()    
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
                
    return n_iter, epoch_cost

def eval_one_epoch(opt, model, test_dataset):
    # Increase num_workers for better data loading
    num_workers = min(8, os.cpu_count())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=num_workers, pin_memory=True,
                                                drop_last=False)   
    epoch_cost = 0
    
    for n_iter,(input_data,label) in enumerate(tqdm(test_loader)):
        # Move data to GPU
        input_data = input_data.cuda()
        label = label.cuda()
        
        suppress_conf = model(input_data)
        
        loss = suppress_loss_func(label,suppress_conf)
        epoch_cost+= loss.detach().cpu().numpy()    
               
    return n_iter, epoch_cost

    
def train(opt): 
    # Setup multi-GPU
    num_gpus = setup_multi_gpu()
    
    writer = SummaryWriter()
    model = SuppressNet(opt)
    
    # Enable multi-GPU training if available
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training")
        model = torch.nn.DataParallel(model)
        # Adjust batch size for multi-GPU
        opt['effective_batch_size'] = opt['batch_size'] * num_gpus
        print(f"Effective batch size: {opt['effective_batch_size']}")
    
    model = model.cuda()
    
    # Initialize best_loss attribute for DataParallel models
    if hasattr(model, 'module'):
        model.module.best_loss = float('inf')
    else:
        model.best_loss = float('inf')
    
    optimizer = optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])     
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])
    
    train_dataset = SuppressDataSet(opt, subset="train")      
    test_dataset = SuppressDataSet(opt, subset=opt['inference_subset'])
    
    for n_epoch in range(opt['epoch']):   
        model.train()
        n_iter, epoch_cost = train_one_epoch(opt, model, train_dataset, optimizer)
            
        writer.add_scalars('sup_data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)
        print("training loss(epoch %d): %f, lr - %f"%(n_epoch,
                                                                epoch_cost/(n_iter+1),
                                                                optimizer.param_groups[0]["lr"]) )
        
        scheduler.step()
        model.eval()
        
        # Use torch.no_grad() for evaluation to save memory
        with torch.no_grad():
            n_iter, eval_cost = eval_one_epoch(opt, model, test_dataset)
        
        writer.add_scalars('sup_data/eval', {'test': eval_cost/(n_iter+1)}, n_epoch)
        print("testing loss(epoch %d): %f"%(n_epoch, eval_cost/(n_iter+1)))
        
        # Handle state dict for DataParallel models
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
            best_loss = model.module.best_loss
        else:
            state_dict = model.state_dict()
            best_loss = model.best_loss
                    
        state = {'epoch': n_epoch + 1,
                'state_dict': state_dict}
        torch.save(state, opt["checkpoint_path"]+"/checkpoint_suppress_"+str(n_epoch+1)+".pth.tar" )
        
        if eval_cost < best_loss:
            if hasattr(model, 'module'):
                model.module.best_loss = eval_cost
            else:
                model.best_loss = eval_cost
            torch.save(state, opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar" )
                
    writer.close()
    return 

def eval_frame(opt, model, dataset):
    # Increase num_workers for better data loading
    num_workers = min(8, os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=num_workers, pin_memory=True,
                                                drop_last=False)
    
    labels_cls={}
    labels_reg={}
    output_cls={}
    output_reg={}                                      
    for video_name in dataset.video_list:
        labels_cls[video_name]=[]
        labels_reg[video_name]=[]
        output_cls[video_name]=[]
        output_reg[video_name]=[]
        
    start_time = time.time()
    total_frames =0  
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0   
    
    for n_iter,(input_data,cls_label,reg_label, _) in enumerate(tqdm(test_loader)):
        # Move data to GPU
        input_data = input_data.cuda()
        cls_label = cls_label.cuda()
        reg_label = reg_label.cuda()
        
        act_cls, act_reg = model(input_data)
        
        cost_reg = 0
        cost_cls = 0
        
        loss = cls_loss_func(cls_label,act_cls)
        cost_cls = loss
            
        epoch_cost_cls+= cost_cls.detach().cpu().numpy()    
               
        loss = regress_loss_func(reg_label,act_reg)
        cost_reg = loss  
        epoch_cost_reg += cost_reg.detach().cpu().numpy()   
        
        cost= opt['alpha']*cost_cls +opt['beta']*cost_reg    
                
        epoch_cost += cost.detach().cpu().numpy() 
        
        act_cls = torch.softmax(act_cls, dim=-1)
        
        total_frames+=input_data.size(0)
        
        for b in range(0,input_data.size(0)):
            video_name, st, ed, data_idx = dataset.inputs[n_iter*opt['batch_size']+b]
            output_cls[video_name]+=[act_cls[b,:].detach().cpu().numpy()]
            output_reg[video_name]+=[act_reg[b,:].detach().cpu().numpy()]
            labels_cls[video_name]+=[cls_label[b,:].detach().cpu().numpy()]
            labels_reg[video_name]+=[reg_label[b,:].detach().cpu().numpy()]
        
    end_time = time.time()
    working_time = end_time-start_time
    
    for video_name in dataset.video_list:
        labels_cls[video_name]=np.stack(labels_cls[video_name], axis=0)
        labels_reg[video_name]=np.stack(labels_reg[video_name], axis=0)
        output_cls[video_name]=np.stack(output_cls[video_name], axis=0)
        output_reg[video_name]=np.stack(output_reg[video_name], axis=0)
    
    cls_loss=epoch_cost_cls/n_iter if n_iter > 0 else 0
    reg_loss=epoch_cost_reg/n_iter if n_iter > 0 else 0
    tot_loss=epoch_cost/n_iter if n_iter > 0 else 0
     
    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames


def test(opt): 
    model = SuppressNet(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/" + opt['exp'] + "ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    
    # Handle DataParallel state dict loading
    if any(key.startswith('module.') for key in base_dict.keys()):
        base_dict = {k.replace('module.', ''): v for k, v in base_dict.items()}
    
    model.load_state_dict(base_dict)
    model = model.cuda()
    model.eval()
    
    dataset = SuppressDataSet(opt, subset=opt['inference_subset'])
    
    # Increase num_workers for better data loading
    num_workers = min(8, os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=num_workers, pin_memory=True,
                                                drop_last=False)   
    labels={}
    output={}                                   
    for video_name in dataset.video_list:
        labels[video_name]=[]
        output[video_name]=[]
        
    with torch.no_grad():
        for n_iter,(input_data,label) in enumerate(test_loader):
            # Move data to GPU
            input_data = input_data.cuda()
            
            suppress_conf = model(input_data)
              
            for b in range(0,input_data.size(0)):
                video_name, idx = dataset.inputs[n_iter*opt['batch_size']+b]
                output[video_name]+=[suppress_conf[b,:].detach().cpu().numpy()]
                labels[video_name]+=[label[b,:].numpy()]
        
    for video_name in dataset.video_list:
        labels[video_name]=np.stack(labels[video_name], axis=0)
        output[video_name]=np.stack(output[video_name], axis=0)

    outfile = h5py.File(opt['suppress_result_file'].format(opt['exp']), 'w')
    
    for video_name in dataset.video_list:
        o=output[video_name]
        l=labels[video_name]
        
        dset_pred = outfile.create_dataset(video_name+'/pred', o.shape, maxshape=o.shape, chunks=True, dtype=np.float32)
        dset_pred[:,:] = o[:,:]  
        dset_label = outfile.create_dataset(video_name+'/label', l.shape, maxshape=l.shape, chunks=True, dtype=np.float32)
        dset_label[:,:] = l[:,:]  
    outfile.close()
    print('complete')


def make_dataset(opt): 
    model = MYNET(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/"+opt['exp']+"_ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    
    # Handle DataParallel state dict loading
    if any(key.startswith('module.') for key in base_dict.keys()):
        base_dict = {k.replace('module.', ''): v for k, v in base_dict.items()}
    
    model.load_state_dict(base_dict)
    model = model.cuda()
    model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    
    with torch.no_grad():
        _, _, _, output_cls, output_reg, labels_cls, labels_reg, _, _ = eval_frame(opt, model, dataset)
    
    proposal_dict=[]
    
    outfile = h5py.File(opt['suppress_label_file'].format(opt['inference_subset']+'_'+opt['setup']), 'w')
    
    num_class = opt["num_of_class"]-1
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                        
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
         
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st, ed]
                    tmp_dict["score"]= cls_anc[anc_idx][label]
                    tmp_dict["label"]=label
                    tmp_dict["gentime"]= idx
                    proposal_anc_dict.append(tmp_dict)
            
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])                
            proposal_dict+=proposal_anc_dict
        
        nms_dict=non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
               
        input_table = np.zeros((duration,unit_size,num_class), dtype=np.float32)
        label_table = np.zeros((duration,num_class), dtype=np.float32)
        
        for proposal in proposal_dict:
            idx = proposal["gentime"]
            conf = proposal["score"]
            cls = proposal["label"]
            for i in range(0,unit_size):
                if idx+i < duration:
                    input_table[idx+i,unit_size-1-i,cls]=conf
        
        for proposal in nms_dict:
            idx = proposal["gentime"]
            cls = proposal["label"]
            label_table[idx:idx+3,cls]=1
        
        dset_input_table = outfile.create_dataset(video_name+'/input', input_table.shape, maxshape=input_table.shape, chunks=True, dtype=np.float32)
        dset_label_table = outfile.create_dataset(video_name+'/label', label_table.shape, maxshape=label_table.shape, chunks=True, dtype=np.float32)
        
        dset_input_table[:]=input_table
        dset_label_table[:]=label_table
        
        proposal_dict=[]
    
    print('complete')
    return
    

def main(opt):
    if opt['mode'] == 'train':
        train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'make':
        make_dataset(opt)
        
    return

if __name__ == '__main__':
    # Set environment variables for better multi-GPU performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/"+opt['exp']+"_opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    
    if opt['seed'] >= 0:
        seed = opt['seed'] 
        torch.manual_seed(seed)
        np.random.seed(seed)
        # For reproducibility in multi-GPU training
        torch.cuda.manual_seed_all(seed)
          
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]  
        
    main(opt)
    while(opt['wterm']):
        pass
