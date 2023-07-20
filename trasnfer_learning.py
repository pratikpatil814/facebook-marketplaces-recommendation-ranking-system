import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime
from sklearn.model_selection import train_test_split
from create_custome_dataset import CustomImageDataset
from tqdm import tqdm
from facebook_resnet50 import facbook_RN_50
from torch.utils.tensorboard import SummaryWriter


def train_model(model,data_loaders,criterion,optimizer,epochs=2):
        # Create a folder for the model evaluation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_eval_folder = f'model_evaluation_loaded_weight/{timestamp}'
    os.makedirs(model_eval_folder, exist_ok=True)

    # Create a folder within model evaluation for the model weights of each epoch
    weights_folder = os.path.join(model_eval_folder, 'weights')
    os.makedirs(weights_folder, exist_ok=True)

    writer_train = SummaryWriter(log_dir='record/train')  
    writer_val = SummaryWriter(log_dir='record/val') 
    for epoch in range(epochs):
        print('Epioch %d / %d'%( epoch,epoch-1))
        print('-'*15)
        device = T.device("cuda" if T.cuda.is_available() else "cpu")

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct = 0
            model = model.to(device)
            pbar = tqdm(data_loaders[phase])

            for inputs , labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                optimizer.zero_grad()
                with T.set_grad_enabled(phase == 'train'):
                    ouputs = model(inputs)
                    loss = F.cross_entropy(ouputs,labels)

                    _,preds = T.max(ouputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                correct += T.sum(preds == labels.data)
                if phase == 'train':
                    writer_train.add_scalar('loss',loss.cpu().item(),epoch)
                else:
                    writer_val.add_scalar('loss', running_loss, epoch)
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = correct.double()/len(data_loaders[phase].dataset)

            print('{} Loss: {:.4f} acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            save_path = os.path.join(model_eval_folder, f'epoch_{epoch+1}.pt')
            T.save(model.state_dict(), save_path)
            metrics = {
                    'epoch': epoch + 1,
                    'validation_loss': running_loss,
                }
            metrics_path = os.path.join(model_eval_folder, f'epoch_{epoch+1}_metrics.pt')
            T.save(metrics, metrics_path)

            writer_train.close() 
            writer_val.close()

if __name__ == '__main__':
    csv_path = './data/training_data/training_data.csv'
    image_path='./data/training_data/new_images_fb'
    dataset = CustomImageDataset(csv_path,img_dir= image_path)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=42)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    val_loader = T.utils.data.DataLoader(val_data, batch_size=2, shuffle=True, num_workers=4)
    data_loaders = {'train': train_loader, 'val': val_loader}

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    model = facbook_RN_50()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr =0.001)
    weights_path = './weights_folder/epoch_1.pt'
    model.load_state_dict(T.load(weights_path))
    train_model(model,data_loaders,criterion,optimizer)
    print(data_loaders)


