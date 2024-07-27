import os
import datetime
from transformers import SamModel, SamConfig, SamProcessor
from ofm import OFM
from torch.utils.data import DataLoader, Subset
from utility import *
from logger import init_logs, get_logger
from dataset import SA1BDataset, MitoDataset, COCOSegmentation
#from coco_dataset import COCOSegmentation
import timeit
from SA1B_NAS_Trainer import SA1B_NAS_Trainer


if __name__ == '__main__':

    DATA_ROOT = '../SAM-finetuner/datasets/' #'/dev/shm/abebe/datasets/' 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()

    NOW = str(datetime.datetime.now()).replace(" ","--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = './logs/{}_dataset[{}]_trainable[{}]_epochs[{}]_lr[{}]_local_bs[{}]/'. \
        format(NOW,args.dataset, args.trainable, args.epochs, args.lr, args.batch_size)
    

    init_logs(log_file_name, args, log_dir)
    args.logger = get_logger()

    #Initialize encoder
    # huge_model = SamModel.from_pretrained("facebook/sam-vit-huge")
    # huge_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # Initialize the original SAM model and processor
    original_model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # OFM configuration and submodel initialization
    elastic_config = {
        "atten_out_space": [512,768], #must be divisbly by num_heads==12
        "inter_hidden_space": [64, 128, 512],
        "residual_hidden_space": [512, 768, 1024],
    }
    elastic_config = {
        "atten_out_space": [768], #Don't go over 768
        "inter_hidden_space": [768,1020,1536], #Reduce for minimizing model size
        "residual_hidden_space": [768,1020,1536],
    }

    ofm = OFM(original_model, elastic_config=elastic_config)

    # saved_supermodel = SamModel.from_pretrained("logs/2024-05-25--01:28:19.953478_dataset[sa1b]_trainable[em]_epochs[100]_lr[1e-05]_local_bs[32]")
    # ofm = OFM(saved_supermodel)

    args.supermodel = ofm
    args.pretrained = original_model

    args.logger.info(f'Original model size : {count_parameters(args.supermodel.model)} params')

    if args.dataset == 'mito':

        # Create an instance of the SAMDataset
        train_dataset = load_dataset("datasets/mitochondria/training.tif", "datasets/mitochondria/training_groundtruth.tif")
        test_dataset = load_dataset("datasets/mitochondria/testing.tif", "datasets/mitochondria/testing_groundtruth.tif")
        train_dataset = MitoDataset(dataset=train_dataset, processor=processor)
        test_dataset = MitoDataset(dataset=test_dataset, processor=processor)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(args.train_subset))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        
        # Apply subset for shorter testing
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.test_subset))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    
    elif args.dataset == 'sa1b':
        if 'e' in args.trainable and 'm' in args.trainable:
            train_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, do_crop=args.crop,label='all_train')
        elif 'e' in args.trainable:
            train_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, do_crop=args.crop,label='from_embedding')
        test_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, do_crop=args.crop,label='all_test')

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = sa1b_collate_fn) #collate_fn = sa1b_collate_fn
        
        else:
            subset_dataset = Subset(train_dataset, indices=range(0,10000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn = sa1b_collate_fn)
        
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = sa1b_collate_fn) #collate_fn = sa1b_collate_fn
        else:
            subset_dataset = Subset(test_dataset, indices=range(10000,len(test_dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn = sa1b_collate_fn)

    elif args.dataset == 'coco':
        #dataset = COCOSegmentation('datasets/coco','val', processor=processor)
        args.base_size = 513
        args.crop_size = 513
        dataset = COCOSegmentation(args,'datasets/coco','val', '2017', processor=processor)
        #test_dataset = COCOSegmentation('datasets/coco','val')

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        
        else:
            subset_dataset = Subset(dataset, indices=range(0,3000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=none_skipper_collate)
        
        if args.test_subset:
            subset_dataset = Subset(dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        else:
            subset_dataset = Subset(dataset, indices=range(3000,len(dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=none_skipper_collate)

    #args.logger.info(f'Dataset : {len(dataset)}')
    args.logger.info(f'trainloader : {len(train_dataloader)} x batch_size : {args.batch_size}')
    args.logger.info(f'testloader : {len(test_dataloader)} x batch_size : {args.batch_size}')

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader
    args.processor = processor
    args.device = device
    args.log_dir = log_dir

    #Reorder mlp layers to see if pretrained model is consistent
    if args.reorder == 'once':
        args.supermodel.mlp_layer_reordering()
    
    #Initialize Trainer
    trainer = SA1B_NAS_Trainer(args)

    # # Calculate IoUs and average IoU before training
    # start_test = timeit.default_timer()
    # miou, mious, map = eval(original_model,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    # end_test = timeit.default_timer()
    # #sorted_mious, indices = torch.sort(mious)
    # #args.logger.info(f'supermodel pre-NAS IoUs: {sorted_mious}')
    # args.logger.info(f'pre-trained model mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    
    start_test = timeit.default_timer()
    #miou, mious, map = eval(args.supermodel.model,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(args.supermodel.model)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'supermodel pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'supermodel pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')

    #save_preds(map,'Largest')
    submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.smallest_model()
    start_test = timeit.default_timer()
    #miou, mious, map = eval(submodel,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(submodel)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'smallest pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'smallest pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    #save_preds(map,'Smallest')
    submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.random_resource_aware_model()
    start_test = timeit.default_timer()
    #miou, mious, map = eval(submodel,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(submodel)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'medium pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'medium pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    #save_preds(map,'Medium')



    args.logger.info(f'NAS Training starts')
    start = timeit.default_timer()

            
    trainer.train()
    
    #train_nas(args)
    
    end = timeit.default_timer()
    args.logger.info(f'NAS Training ends : {round(end-start,4)} seconds')