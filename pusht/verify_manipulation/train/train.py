import torch
from pusht.verify_manipulation.train.network_score import NeuralNetwork, TrajectoryDataset
import sys
from torch import nn
from tqdm.auto import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Initialize model
model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.MSELoss() # MSE Loss for risk evaluation

#optimizer = torch.optim.SGD(lr=5e-4,params=model.parameters()) # Create an optimizer
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=1e-6, 
                             weight_decay=1e-2)
#Create dataset
#filepath1 = '/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_400.pkl'
#filepath2 = '/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_3200.pkl'
filepaths = ['/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/l_100/pusht_train_'+str(i+1)+'.pkl' for i in range(10)]
dataset = TrajectoryDataset(filepaths)
train_size = int(1.0 * len(dataset))
# create dataloader
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1024,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)
# visualize data in batch
batch = next(iter(train_dataloader))
print("size of batched dataset:", len(train_dataloader))
print("seed:", batch['seed'].shape)
print("theta", batch['theta'].shape)
print(batch['score'].shape)

#torch.manual_seed(42)

# Set the number of epochs
num_epochs = 500

# Build training and evaluation loop
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    epoch_loss = list()
    loss_list = []
    for epoch_idx in tglobal:
        for nbatch in iter(train_dataloader):
        ### Training
          model.train()
          #X1_train = nbatch['theta'].to(device).float().reshape((-1,1))
          X2_train = nbatch['seed'].to(device).float()

          score_train = nbatch['score'].to(device).float()
          #X_input = torch.hstack((X1_train,X2_train))#concatenate X=[theta,x,y]
          #print(X_input.shape)
          # 1. Forward pass 
          
          score_pred = model(X2_train).float() #.squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
          #print('score_est',score_pred,'score_pred',score_train)
          loss = loss_fn(score_pred, # Using nn.MSELoss 
                        score_train)
          #print(X_input,score_pred,score_train)
          # 3. Optimizer zero grad
          optimizer.zero_grad()

          # 4. Loss backwards
          loss.backward()
          loss_list.append(loss.detach().cpu())
          # 5. Optimizer step
          optimizer.step()

          ### Testing
          model.eval()
          if loss<0.05:
              break
      # Print out what's happening every 10 epochs
        if epoch_idx % 10 == 0:
            print(f"Epoch: {epoch_idx} | Loss: {loss:.5f}")# | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Save information
EPOCH = 5
PATH = "data/model_mlp_relu.pt"
LOSS = loss

torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)