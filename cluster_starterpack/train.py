import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
import wandb

def print_results(results, dataset, wandb_run, epoch):
    
    print("RESULTS: ")
    print("  Action class accuracy: {:.3f} %".format(results["accuracy_action"]))
    print("  Offence severity accuracy:  {:.3f} %".format(results["accuracy_offence_severity"]))

    # Log the statistics to Wandb 
    # Test set -> Summary statistic
    if (dataset == "Test"):
        wandb_run.summary["Test_acc_action"] = round(results["accuracy_action"], 3)
        wandb_run.summary["Test_acc_offense_severity"] = round(results["accuracy_offence_severity"], 3)

    # Train & Validation sets -> Plots    
    else:
        wandb_run.log({"Epoch": epoch, 
                    f"{dataset}_acc_action": round(results["accuracy_action"], 3), 
                    f"{dataset}_acc_offense_severity": round(results["accuracy_offence_severity"], 3)}
                    )

    return


def set_wandb_metrics(wandb_run):

    wandb_run.define_metric(name="Epoch", hidden=True) # Don't plot the Epoch metric

    wandb_run.define_metric(name="Train_acc_action", summary="max", step_metric="Epoch")
    wandb_run.define_metric(name="Valid_acc_action", summary="max", step_metric="Epoch")

    wandb_run.define_metric(name="Train_acc_offense_severity", summary="max", step_metric="Epoch")
    wandb_run.define_metric(name="Valid_acc_offense_severity", summary="max", step_metric="Epoch")


def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            model_saving_dir,
            epoch_start,
            model_name,
            path_dataset,
            wandb_run,
            model_artifact,
            max_epochs=1000
            ):
    

    set_wandb_metrics(wandb_run)

    for epoch in range(epoch_start, max_epochs+1): # [epoch_start, max_epoch]
        
        print(f"######################  Epoch {epoch}/{max_epochs} ###################### ")
    
        print("###################### TRAINING ###################")
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )
        pbar.close()

        results_train = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print_results(results_train, "Train", wandb_run, epoch)

        print("###################### VALIDATION ###################")
        pbar = tqdm(total=len(val_loader2), desc="Validation", position=0, leave=True)
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid",
            pbar=pbar
        )
        pbar.close()

        results_val = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print_results(results_val, "Valid", wandb_run, epoch)

        scheduler.step()

        # Save the model every 4 epochs
        if (epoch % 4 == 0):
            state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            print("***** SAVING MODEL *****")
            path_aux = os.path.join(model_saving_dir, str(epoch) + "_model.pth.tar")
            torch.save(state, path_aux) # This saves the state_dict info locally
            if (model_artifact != None):
                model_artifact.add_file(path_aux)


    # Evaluate on the Test set after training
    print("###################### TEST ###################")
    pbar = tqdm(total=len(test_loader2), desc="Test", position=0, leave=True)
    prediction_file, loss_action, loss_offence_severity = train(
            test_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="test",
            pbar=pbar
        )
    pbar.close()

    results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
    print_results(results, "Test", wandb_run, epoch)
    

    # Save the final model if not already done
    if (max_epochs % 4 != 0):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        print("***** SAVING MODEL *****")
        path_aux = os.path.join(model_saving_dir, str(max_epochs) + "_model.pth.tar")
        torch.save(state, path_aux) # This saves the state_dict info locally
        if (model_artifact != None):
            model_artifact.add_file(path_aux)

    if (model_artifact != None):
        wandb_run.log_artifact(model_artifact)
    print("###################### TRAINER DONE ###################")

    return


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            
            if pbar is not None:
                pbar.update()

            # compute output
            if (train):
                outputs_offence_severity, outputs_action, _ = model(mvclips)
            else:
                with torch.no_grad():
                    outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
          
        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss




# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):
    

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
           
    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            #mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values                    


        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file
