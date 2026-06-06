import torch
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.MSELoss()




def train_single(num_epochs, num_batches,batch_size, model, optimizer, replay_buffer):
    for epoch in range(num_epochs):

        for i in range(num_batches):
            optimizer.zero_grad()
            t1_observations, t1_actions, _, t1_next_observations, _ = replay_buffer.sample(batch_size)
            oa_in = torch.cat([t1_observations, t1_actions], dim=-1)

            next_o_pred = model(oa_in)
            loss = loss_fn(next_o_pred, t1_next_observations)

            loss.backward()
            optimizer.step()


def train_model(model, replay_buffer, optimizer, num_epochs=500, batch_size=32):
    """
    Train a single model with supervised learning
    """
    idxs = np.array(range(len(replay_buffer)))
    num_batches = len(idxs) // batch_size
    if not isinstance(model, list):
        train_single(num_epochs, num_batches, batch_size, model, optimizer, replay_buffer)
    else:   
        #========== TODO: start ==========
        # Write code to train the ensemble of models.
        # Hint1: Each model should have a different batch size for each model
        # Hint2: check out how we define optimizer and model for ensemble models. 
        # During training, each model should have their individual optimizer to increase diversity.
        # Hint3: You can use the train_single function to train each model.
        pass 
        #========== TODO: end ==========
        
        
        

