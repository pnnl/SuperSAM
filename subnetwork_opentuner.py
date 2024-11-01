import numpy as np
from ofm import OFM
import torch, time, opentuner
from opentuner import ConfigurationManipulator, EnumParameter, MeasurementInterface, Result
from typing import List
from dataset import SA1BDataset
from torch.utils.data import DataLoader, Subset
from transformers import SamModel, SamConfig, SamProcessor
from utility import sa1b_collate_fn, count_parameters, get_args, compute_iou
import torch.nn as nn


def eval(subnetwork,test_dataloader,device='cuda',prompt='p'):
    mIoU = []
    model = model.eval()
    model = nn.DataParallel(model).to(device)
    
    for inputs in test_dataloader: #enumerate(tqdm(self.test_dataloader, disable=self.no_verbose)): 

        torch.cuda.empty_cache()

        # print(f'inputs["pixel_values"] : {inputs["pixel_values"].shape}')
        # print(f'inputs["input_points"] : {inputs["input_points"].shape}')
        # print(f'inputs["input_boxes"] : {inputs["input_boxes"].shape}')
        # print(f'inputs["ground_truth_masks"] : {inputs["ground_truth_masks"].shape}')

        if len(inputs["input_points"].shape) > 4:
            inputs["input_points"] = inputs["input_points"].squeeze((2))

        with torch.no_grad():
            if prompt == 'p':
                outputs = model(pixel_values=inputs["pixel_values"].to(device),
                                input_points=inputs["input_points"].to(device),
                                multimask_output=True)
            elif prompt == 'b':
                outputs = model(pixel_values=inputs["pixel_values"].to(device),
                                input_boxes=inputs["input_boxes"].to(device),
                                multimask_output=True)

        
        outs = outputs.pred_masks
        scores = outputs.iou_scores

        #loop through batch (images)
        for i, one_output in enumerate(outs):
            #loop through objects
            for j, preds in enumerate(one_output):
                pt, bx = inputs["points"][i][j], inputs["boxes"][i][j]

                #loop through objects
                pred_1,pred_2,pred_3 = torch.sigmoid(preds[0]),torch.sigmoid(preds[1]),torch.sigmoid(preds[2])
                pred_1,pred_2,pred_3 = (pred_1 > 0.5),(pred_2 > 0.5),(pred_3 > 0.5)

                gt = inputs["ground_truth_masks"][i][j]

                _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])

                mIoU.append(max(mious))

                #Record ground truth with prediction
                if map != None:
                    img_idx = inputs["img_id"][i]
                    map[f'img-{img_idx}-obj-{j}-pred-0'] = (inputs["image"][i],pt,bx,gt,pred_1,mious[0])
                    map[f'img-{img_idx}-obj-{j}-pred-1'] = (inputs["image"][i],pt,bx,gt,pred_2,mious[1])
                    map[f'img-{img_idx}-obj-{j}-pred-2'] = (inputs["image"][i],pt,bx,gt,pred_3,mious[2])

    model = model.module
    model = model.to('cpu')
    mIoU = torch.tensor(mIoU)
    average_mIoU = torch.mean(mIoU).item()
    return average_mIoU, mIoU, map

DATA_ROOT = '../SAM-finetuner/datasets/' #'/dev/shm/abebe/datasets/' 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#torch.cuda.empty_cache()
args = get_args()


DATA_ROOT = 'datasets/'
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


test_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, do_crop=False,label='all_test')
subset_dataset = Subset(test_dataset, indices=range(10000,10100,1))
test_dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn = sa1b_collate_fn) #collate_fn = sa1b_collate_fn
       

saved_supermodel = SamModel.from_pretrained('logs/2024-09-19--00:00:49.670661_dataset[sa1b]_trainable[em]_epochs[2]_lr[1e-05]_local_bs[16]/Best')
ofm = OFM(saved_supermodel)
supermodel = ofm.model



print("Original FM number of parameters:",ofm.total_params)

# Experiment
## Randomly sample a downsize model
ds_model,_,_ = ofm.random_resource_aware_model()
## Check the downsize model architecture

# We are not do binary-masking or zero-masking the weights, we etract the clean downsize model, and the model size **is truely reduced**.

params = count_parameters(ds_model)
print(f'Params : {params}')
print(f'config : {ds_model.config.elastic_config}')
print(f'num_layers : {ds_model.vision_encoder.config.num_hidden_layers}')


def arc_config_creator(
    atten_out_space: List[int],
    inter_hidden_space: List[int],
    residual_hidden_space: List[int],
    layer_elastic:List[float],
    n_layer=12,
    smallest=False,
    largest=False,
) -> dict:
    """Generate subnet architecture configuration based on the provided configuration.

    Args:
        atten_out_space (list[int]): Attention head output hidden size space, NOT the hidden space.
        inter_hidden_space (list[int]): Intermediate dense hidden layer size space.
        residual_hidden_space (list[int]): Attention (input size) and Intermediate layer (out size) hidden size.
        n_layer (int, optional): Number of multi-head attention layers. Defaults to 12.
        smallest (bool, optional): Either return smallest subnet configuration. Defaults to False.

    Returns:
        dic: Subnet architecture configure.
    """
    arc_config = {}
    np.random.seed(int(time.time()))  # Set the seed to the current time
    residual_hidden = np.random.choice(residual_hidden_space).item()
    
    assert smallest == False or largest == False  # Only one can be true

    search_space = {}

    for layer in range(n_layer):

        conf_key = "layer_" + str(layer+1) + "_atten_out"
        search_space[conf_key] = atten_out_space
        
        conf_key = "layer_" + str(layer+1) + "_inter_hidden"
        search_space[conf_key] = inter_hidden_space

        conf_key = "layer_" + str(layer+1) + "_residual_hidden"
        search_space[conf_key] = residual_hidden_space
        
    return search_space

search_space = arc_config_creator(atten_out_space=supermodel.config.elastic_config['atten_out_space'], residual_hidden_space=supermodel.config.elastic_config['residual_hidden_space'], inter_hidden_space=supermodel.config.elastic_config['inter_hidden_space'], n_layer=supermodel.vision_encoder.config.num_hidden_layers)


class NASOpenTuner(MeasurementInterface):


    def manipulator(self):
        """Generate subnet architecture configuration based on the provided configuration.

        Args:
            atten_out_space (list[int]): Attention head output hidden size space, NOT the hidden space.
            inter_hidden_space (list[int]): Intermediate dense hidden layer size space.
            residual_hidden_space (list[int]): Attention (input size) and Intermediate layer (out size) hidden size.
            n_layer (int, optional): Number of multi-head attention layers. Defaults to 12.
            smallest (bool, optional): Either return smallest subnet configuration. Defaults to False.

        Returns:
            dic: Subnet architecture configure.
        """

        manipulator = ConfigurationManipulator()

        for key in search_space:
            manipulator.add_parameter(
                EnumParameter(key, search_space[key]))
        
        # manipulator.add_parameter(
        #         EnumParameter("residual_hidden", search_space[key]))
        
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """

        cfg = desired_result.configuration.data

        arc_config = {}
        for layer in range(supermodel.vision_encoder.config.num_hidden_layers):
            attn_key = "layer_" + str(layer+1) + "_atten_out"
            inter_key = "layer_" + str(layer+1) + "_inter_hidden"
            residual_key = "layer_" + str(layer+1) + "_residual_hidden"
            
            arc_config[f"layer_{layer + 1}"] = {
                "atten_out": cfg[attn_key],
                "inter_hidden": cfg[inter_key],
                "residual_hidden": cfg[residual_key],
            }
        
        subnetwork, total_params = ofm.resource_aware_model(arc_config)

        ## Now we use Trainer to evaluate the zero-shot downsized model

        average_mIoU, mIoU, map = eval(subnetwork,test_dataloader,device='cuda', prompt='p')

        print('Subnetwork')
        print(f'\tConfig {arc_config}')
        print(f'\tParams : {total_params}')
        print(f'\tmIoU : {average_mIoU}')
        
        return Result(time=(1-average_mIoU))
    
    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal configurations written to NAS_final_config.json:", configuration.data)
        self.manipulator().save_to_file(configuration.data,
                                        'SAMNAS_final_config.json')


argparser = opentuner.default_argparser()
start = time.time()
NASOpenTuner.main(argparser.parse_args())
stop = time.time()
