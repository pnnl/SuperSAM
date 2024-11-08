# SuperSAM: Crafting a SAM supernetwork via structured pruning and unstructured parameter prioritization
This is the official implementation for the paper:


## Usage

Train supernet on SA1B dataset.

```bash
python nas.py --dataset=sa1b \
--epochs=5 --batch_size=8 --trainable=em \
--lr=1e-5 --weight_decay=0 --train_subset=400 \
--test_subset=100 --train_prompt=p \
--test_prompt=p --loss=dice --sandwich=lsm \
--no_verbose --save_interval=4 \
```


Anonymous

<!-- ## TODO

- [x] ViT pre-trained ckpts
- [x] ViT FL simulation scripts
- [x] Tensorboard logger
- [x] Elastic space APIs for system-heteo
- [x] Load ckpt high-level APIs
- [x] Simulation scripts on GLUE
- [x] ViT CIFAR-100 ckpts
- [x] High level API for real edge-FL
- [x] API for segment anything (SAM)
- [x] Evaluate Scripts for resource-aware models
- [ ] BERT-large, FLAN-T5 ckpts
- [ ] Simulation scripts on SQUAD
- [ ] ONNX and TensorRT APIs for edge
- [ ] Tiny fedlib -->

## Citation

If you find our work is helpful, please kindly support our efforts by citing our paper:

```

under review

```

## Acknowledgement

The experiments of this work is sponsored by **[anonymous institution]** and **[anonymous institution]**.

```

```
