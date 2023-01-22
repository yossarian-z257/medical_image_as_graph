import os
import sys
import argparse
from utils import graph_preperation
from utils import str_to_bool


current_file_path = os.path.dirname(os.path.abspath(__file__))


def main(model_name = 'densenet', superpixel_number = 10, learning_rate = 0.001, batch_size = 64, epochs = 10, train = True, saved = True):#, pretrained = False
    print("Model name: ", model_name)

    if saved:
        print("Using saved state")
        if train:
            print("Training")
        else:
            print("Testing")
    else:
        print("Not using saved state")
        graph_preperation(superpixel_number,model_name)
    
    

















if __name__ == "__main__":
    #argument parsing
    #check if file exists
    if not os.path.exists(f'{current_file_path}/chest_xray'):
        print("please make sure that chest_xray folder is in the same directory as main.py")
        print("expected chest_xray folder with train and test folders inside")
        print("run dowload_data.py to download the data and extract it to the chest_xray folder")
        sys.exit()
    parser = argparse.ArgumentParser(description='Getting arguments')
    parser.add_argument('--model_name', type=str, default='densenet121', help='Name of the model')
    parser.add_argument('--superpixel_number', type=int, default=10, help='Number of superpixels')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train', type=bool, default=True, help='Train or test')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_saved_state', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    print(args)

    model = args.model_name
    superpixel_number = args.superpixel_number
    learning_rate = args.learning_rate
    train = args.train
    batch_size = args.batch_size
    epochs = args.epochs
    saved = args.use_saved_state
    #pretrained = args.pretrained
    print("starting main.py",saved)

    main(model, superpixel_number, learning_rate, batch_size, epochs, train, saved)#, pretrained

    # print(args.model_name)
    # print(args.superpixel_number)
    # print(args.learning_rate)
