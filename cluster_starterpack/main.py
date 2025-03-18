import os
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from dataset import MultiViewDataset
from train import trainer, evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from model import MVNetwork
from torchvision.models.video import R3D_18_Weights, R2Plus1D_18_Weights, MViT_V2_S_Weights, Swin3D_T_Weights
from torchvision.models.video import MC3_18_Weights, S3D_Weights
import wandb


def checkArguments(args):

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean' and args.pooling_type != 'attention':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()

    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or  args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or  args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or  args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()

    # args.pre_model
    if args.pre_model not in ["r3d_18", "r2plus1d_18", "mvit_v2_s", "swin3d_t", "mc3_18", "s3d"]:
        print("Could not find the desired pretrained model")
        print("Possible options are: r3d_18, r2plus1d_18, mvit_v2_s, swin3d_t, mc3_18, s3d")
        exit()

    # args.only_evaluation
    if args.only_evaluation not in [0,1,2,3]:
        print("Invalid task option (only_evaluation)")
        print("Possible arguments are: 0, 1, 2, 3")
        exit()


def main(args, wandb_run, model_artifact):

    # Retrieve the script argument values
    if args:
        lr = args.LR
        gamma = args.gamma
        step_size = args.step_size
        start_frame = args.start_frame
        end_frame = args.end_frame
        weight_decay = args.weight_decay
        model_name = args.model_name
        pre_model = args.pre_model
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        pooling_type = args.pooling_type
        weighted_loss = args.weighted_loss
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
    else:
        print("ERROR: No arguments given.")
        exit()


    model_saving_dir = os.path.join("models", os.path.join(model_name))
    os.makedirs(model_saving_dir, exist_ok=True)



    # Initialize the data augmentation, only used for the training data
    # Apply random transformations to improve generalization
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                          transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                          transforms.RandomRotation(degrees=5),
                                          transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                          transforms.RandomHorizontalFlip()
                                          ])
    else:
        transformAug = None

    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()         # .transforms(): returns a set of data preprocessing transformations to prepare input     
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    elif pre_model == "swin3d_t":
        transforms_model = Swin3D_T_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "s3d":
         transforms_model = S3D_Weights.KINETICS400_V1.transforms()
    

    # Create only the relevant Datasets and DataLoaders for this task
    if only_evaluation == 0:
        print("--> ONLY Evaluating on the test set")
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
                                         transform_model=transforms_model)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2, 
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
        
    elif only_evaluation == 1:
        print("--> ONLY Evaluating on the challenge set")
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views = 5, 
                                         transform_model=transforms_model)

        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall, 
                                                    batch_size=1, shuffle=False,
                                                    num_workers=max_num_worker, pin_memory=True)
        
    elif only_evaluation == 2:
        print("--> ONLY Evaluating on the test and challenge set")
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
                                         transform_model=transforms_model)
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views = 5, 
                                         transform_model=transforms_model)

        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
        
        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
                                                    batch_size=1, shuffle=False,
                                                    num_workers=max_num_worker, pin_memory=True)
    else:
        print(f"--> Training and validation for {max_epochs} epcohs, and then evaluation on the test")

        # Create Train Validation and Test datasets
        dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
                                         num_views=num_views, transform=transformAug, transform_model=transforms_model)
        dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views = 5, 
                                          transform_model=transforms_model)
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
                                         transform_model=transforms_model)

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train, 
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)

    print(f"--> Creating the model: {pre_model} with pooling: {pooling_type}")
    model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()

    if path_to_model_weights != "":
        print("--> Loading model weights from: ", path_to_model_weights)
        path_model = os.path.join(path_to_model_weights)
        load = torch.load(path_model)
        model.load_state_dict(load['state_dict'])


    # Set up training parameters if training
    if only_evaluation == 3:
        print("--> Optimizer: AdamW")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr, 
                                      betas=(0.9, 0.999),
                                      eps=1e-07, 
                                      weight_decay=weight_decay,
                                      amsgrad=False)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        epoch_start = 1

        if continue_training:
            print("--> Conitnuing training from: ", model_saving_dir, "/model.pth.tar")
            path_model = os.path.join(model_saving_dir, 'model.pth.tar')
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])
            optimizer.load_state_dict(load['optimizer'])
            scheduler.load_state_dict(load['scheduler'])
            epoch_start = load['epoch']

        print("--> Epoch Start: ", epoch_start)

        if weighted_loss == 'Yes':
            print("--> Weighted loss")
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        else:
            print("--> NO Weighted loss")
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            criterion = [criterion_offence_severity, criterion_action]


    # Start training or evaluation
    if only_evaluation == 0:
        print("--> Starting Evaluation (Test set)...")
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
        ) 
        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 1:
        print("--> Starting Evaluation (Challenge set)...")
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)

    elif only_evaluation == 2:
        print("--> Starting Evaluation (Challenge set)...")
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
        )

        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        print("--> Starting Evaluation (Challenge set)...")
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)
    else:
        print("--> Starting Trainer...")
        trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, 
                model_saving_dir, epoch_start, model_name=model_name, path_dataset=path, wandb_run=wandb_run,
                model_artifact=model_artifact, max_epochs=max_epochs)
        
    print("--> MAIN DONE! ")
    return 0


if __name__ == '__main__':

    print("################################################################################")
    print(" ---------------------------- DATA KICKERS FC ----------------------------------")
    print()

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=60,     help='Maximum number of epochs' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VARS",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="r2plus1d_18", help="Name of the pretrained model")
    parser.add_argument("--pooling_type", required=False, type=str, default="max", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3, help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to the model weights")

    parser.add_argument("--wandb_run_name", required=True, type=str, help="Wandb run name")
    parser.add_argument("--wandb_saving_model_name", required=False, type=str, default="", help="Name of the Artifact to save the checkpoints in")

    args = parser.parse_args()

    ## Checking if arguments are valid
    checkArguments(args)

    # Initialize Wandb
    wandb_run = wandb.init(project="IFT6759_MVFoulR", 
                           name=args.wandb_run_name,
                           config= {"Pre-Trained model": args.pre_model,
                                    "Pooling type": args.pooling_type,
                                    "Batch size": args.batch_size,
                                    "Learning rate": args.LR,
                                    "Max epochs": args.max_epochs,
                                    "Data augmentation": args.data_aug,
                                    "Number of views": args.num_views,
                                    "FPS": args.fps}
                            )
    
    if (args.wandb_saving_model_name != ""):
        model_artifact = wandb.Artifact(name=args.wandb_saving_model_name,
                                        type="model")
    else:
        model_artifact = None


    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    print('Starting main function')
    main(args, wandb_run, model_artifact)
    print(f'Total Execution Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-start))}')
    wandb_run.finish()
