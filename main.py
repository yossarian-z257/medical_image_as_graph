import os
import sys
import argparse
from utils import graph_preperation
from utils import str_to_bool
from utils import dataloader, get_gnn_model, save_model, save_plots, SaveBestModel
from config import *
import torch
from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from train import train_epoch, valid_epoch

current_file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_epoch(model, train_loader,val_loader, optimizer, criterion, epoch = 10, train = True, cnn_model_name = 'densenet121', gnn_model = 'GCN', superpixel_number = 10):
    epoch_loss = 0
    epoch_acc = 0
    history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}
    save_best_model = SaveBestModel()
    if train:
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model,device,train_loader,criterion,optimizer)
            val_loss, val_acc = valid_epoch(model,device,val_loader,criterion)
  
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f} AVG Training Acc {:.4f} % AVG Val Acc {:.4f} %".format(epoch + 1,
                                                                                                                    10,
                                                                                                                    train_loss,
                                                                                                                    val_loss,
                                                                                                                    train_acc,
                                                                                                                    val_acc))
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            save_best_model(val_loss, epoch, model, optimizer, criterion, cnn_model_name, gnn_model, superpixel_number)



    else:
        test_loss, test_acc = valid_epoch(model,device,val_loader,criterion)

    save_plots(history['train_acc'], history['val_acc'], history['train_loss'], history['val_loss'],cnn_model_name, gnn_model, superpixel_number)
    



def main(cnn_model_name = 'densenet', gnn_model = 'GCN', superpixel_number = 10, learning_rate = 0.001, batch_size = 64, epochs = 10, train = True, saved = True):#, pretrained = False
    print("Model name: ", cnn_model_name)

    criterion = torch.nn.CrossEntropyLoss()
    if saved:
        print("Using saved state")
        if train:
            print("Training")
            train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader(FOLDER_PATH , batch_size, saved, superpixel_number, cnn_model_name)
            train_dataset = [data.to(device) for data in train_dataset]
            val_dataset = [data.to(device) for data in val_dataset]
            test_dataset = [data.to(device) for data in test_dataset]
            # dataset = ConcatDataset([train_dataset, val_dataset])
            model = get_gnn_model(gnn_model, feature_size[cnn_model_name])
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
            run_epoch(model, train_loader, test_loader, optimizer, criterion, epochs, train, cnn_model_name, gnn_model, superpixel_number)
            print("training complete")


        else:
            print("Testing")
            print(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_best_model.pt')
            _ , test_loader, _, test_dataset, _ , _  = dataloader(FOLDER_PATH , batch_size, saved, superpixel_number, cnn_model_name)
            model = get_gnn_model(gnn_model, feature_size[cnn_model_name])
            model.load_state_dict(torch.load(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_best_model.pt'))
            test_loss, test_acc = valid_epoch(model,device,test_loader,criterion)
            print("Test Loss: {:.4f}".format(test_loss))
            print("Test Acc: {:.4f} %".format(test_acc))
            print("testing complete")
    
    else:
        print("Not using saved state")
        graph_preperation(superpixel_number,cnn_model_name)
        saved = True
        main(cnn_model_name, gnn_model, superpixel_number, learning_rate, batch_size, epochs, train, saved)
        return 0


    return 0 
















if __name__ == "__main__":


    if not os.path.exists(f'{current_file_path}/chest_xray'):
        print("please make sure that chest_xray folder is in the same directory as main.py")
        print("expected chest_xray folder with train and test folders inside")
        print("run dowload_data.py to download the data and extract it to the chest_xray folder")
        sys.exit()
    parser = argparse.ArgumentParser(description='Getting arguments')
    parser.add_argument('--cnn_model_name', type=str, default='densenet121', help='Name of the model')
    parser.add_argument('--gnn_model', type=str, default='GCN', help='Name of the GNN model')
    parser.add_argument('--superpixel_number', type=int, default=10, help='Number of superpixels')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train', type=str_to_bool, nargs='?', const=True, default=True, help='Train or test')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_saved_state', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    cnn_model_name = args.cnn_model_name
    gnn_model = args.gnn_model
    superpixel_number = args.superpixel_number
    learning_rate = args.learning_rate
    train = args.train
    batch_size = args.batch_size
    epochs = args.epochs
    saved = args.use_saved_state
    print(f"starting main.py for saved state = {saved}")

    ret = main(cnn_model_name,gnn_model, superpixel_number, learning_rate, batch_size, epochs, train, saved)#, pretrained
    if ret == 0:
        print("Done")
    else:
        print("Error")

