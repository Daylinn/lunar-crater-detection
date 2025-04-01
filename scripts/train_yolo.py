"""
My lunar crater detection training script using YOLOv5.
This is where the magic happens - we train our model to spot craters on the moon!
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add the yolov5 directory to our Python path so we can use its modules
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov5.train import train
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import LOGGER, TQDM, check_git_status, check_requirements, increment_path
from yolov5.utils.torch_utils import select_device, smart_inference_mode

def create_yolo_config():
    """Create YOLO configuration file"""
    config = {
        'path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),  # dataset root dir
        'train': 'train/images',  # train images (relative to 'path')
        'val': 'valid/images',    # val images (relative to 'path')
        'test': 'test/images',    # test images (relative to 'path')

        'names': {0: 'crater'},   # class names
        'nc': 1,                  # number of classes

        # Training parameters - optimized for faster training
        'epochs': 50,             # reduced from 100 to 50
        'batch_size': 32,         # increased from 16 to 32
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model parameters - using smaller model for faster training
        'model': 'yolov5n.pt',    # changed from yolov5s.pt to yolov5n.pt (nano model)
        'pretrained': True,       # use pretrained model
        'optimizer': 'SGD',       # optimizer
        'lr0': 0.01,             # initial learning rate
        'momentum': 0.937,       # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'warmup_epochs': 2,      # reduced from 3 to 2
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,   # warmup initial bias lr
        'box': 7.5,              # box loss gain
        'cls': 0.5,              # cls loss gain
        'dfl': 1.5,              # dfl loss gain

        # Additional optimizations
        'cache': True,           # cache images in memory
        'workers': 4,            # reduced from 8 to 4 for CPU
        'amp': True,             # mixed precision training
    }
    
    # Save configuration
    config_path = os.path.join(os.path.dirname(__file__), 'yolo_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config

def train_model(config):
    """Train YOLOv5 model"""
    # Initialize model
    model = YOLO(config['model'])
    
    # Train the model
    results = model.train(
        data=os.path.join(os.path.dirname(__file__), 'yolo_config.yaml'),
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['imgsz'],
        device=config['device'],
        pretrained=config['pretrained'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        box=config['box'],
        cls=config['cls'],
        dfl=config['dfl'],
        cache=config['cache'],
        workers=config['workers'],
        amp=config['amp']
    )
    
    return results

def validate_model(model_path, data_yaml):
    """Validate trained model"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results

def predict_image(model_path, image_path, conf_threshold=0.25):
    """Run inference on a single image"""
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        show=True
    )
    return results

def main(opt):
    # Save our training settings
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, project, name, exist_ok, pretrained, optimizer, verbose, seed, deterministic, local_rank = opt

    # Set up our training directory
    save_dir = Path(save_dir)
    device = select_device(local_rank)

    # Load our dataset configuration
    data = check_file(data)  # check file
    with open(data, encoding='ascii', errors='ignore') as f:
        data = yaml.safe_load(f)  # dictionary

    # Set up our training paths
    train_path = data['train']
    test_path = data['val']  # we'll use validation set for testing

    # Figure out if we're doing single-class or multi-class detection
    if single_cls:
        data['names'] = ['crater']  # force single-class mode

    # Set up our model configuration
    cfg = check_file(cfg)  # check file
    with open(cfg, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # dictionary

    # Set up our training parameters
    data['train'] = str(train_path)  # force to string
    data['val'] = str(test_path)  # force to string

    # Set up our model weights
    weights = str(weights).strip()  # strip to get clean filename
    if not weights:
        weights = 'yolov5s.pt'  # default to small model
        pretrained = True
    weights = check_file(weights)  # check file
    if not pretrained and not weights.endswith('.pt'):  # check if weights is a local file
        weights = check_file(weights)  # check file

    # Set up our training directory
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Set up our training parameters
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'  # use CUDA
    init_seeds(seed + 1 + RANK, deterministic=deterministic)  # set random seeds

    # Set up our data loaders
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if none
    train_path, val_path = data_dict['train'], data_dict['val']

    # Set up our model
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Set up our optimizer
    optimizer = smart_optimizer(model, optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Set up our scheduler
    scheduler = smart_scheduler(optimizer, max_epochs=epochs, warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr, end_epoch=hyp['cos_lr'] * epochs, power=hyp['power'])  # parameterized scheduler

    # Set up our training callbacks
    callbacks = Callbacks()
    if plots:
        callbacks.add_action('on_train_end', on_train_end)

    # Set up our training loop
    start_epoch, best_fitness = 0, 0.0
    if ckpt is not None:
        start_epoch, best_fitness = resume_training(ckpt, model, optimizer, epoch, fitness)
    if resume:
        start_epoch, best_fitness = resume_training(ckpt, model, optimizer, epoch, fitness)

    # Set up our training metrics
    callbacks.run('on_pretrain_routine_end')

    # Start training!
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # create scaler
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (nc > 1) + 1.  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(enumerate(train_loader), total=nb, desc=f'Epoch {epoch}/{epochs - 1}')  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float16, 0-255 to 0-1

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # scale loss by world size
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()  # update scaler
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                if i % log_imgs == 0 or (epoch == epochs - 1 and i == nb - 1):  # Batch log
                    mloss = (loss_items * batch_size).mean() * WORLD_SIZE  # mean losses
                    macc = torch.stack(metrics['accuracy']).mean() * WORLD_SIZE  # mean accuracy
                    speed = f'{batch_size * WORLD_SIZE / dt[3]:.1f}imgs'  # speed
                    if epoch == epochs - 1 and i == nb - 1:
                        pbar.desc = f"{mloss:.3f} {macc:.3f} {speed}"
                    callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, vis_batch)
                    if callbacks.stop_training:
                        return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                             batch_size=batch_size // WORLD_SIZE * 2,
                                             imgsz=imgsz,
                                             model=ema.ema,
                                             single_cls=single_cls,
                                             dataloader=val_loader,
                                             save_dir=save_dir,
                                             plots=False,
                                             callbacks=callbacks,
                                             compute_loss=compute_loss)  # val best model with plots

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, src=0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        save_json(f, results)  # save best.pt under the same name with .json suffix

        callbacks.run('on_train_end', last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 