import torch
from tqdm import tqdm
import os
import argparse
import random
from torch.utils.tensorboard import SummaryWriter

from models.linear_models import (
    VectorQuantizer,
    Encoder,
    Decoder,
    VQVAEModel,
)
from utils import load_dataset, set_seed_everywhere, MAX_EPISODES

def get_args():
    parser = argparse.ArgumentParser()

    default_env = 'Phoenix'

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default=default_env, type=str)
    parser.add_argument("--datapath", default="dataset", type=str)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--train_type", default="confounded", type=str, choices=["normal", "confounded"])

    # Save & Evaluation
    parser.add_argument("--num_episodes", default=MAX_EPISODES[default_env], type=int)
    parser.add_argument("--n_epochs", default=1000, type=int)

    # VQVAE & Hyperparams
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_embeddings", default=512, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    parser.add_argument("--commitment_cost", default=0.25, type=float)
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--val_split_ratio", default=0.95, type=float)

    args = parser.parse_args()
    return args

def train(args):
    
    set_seed_everywhere(args.seed)
   
    observations, _, _, _ = load_dataset(
        args.env,
        args.seed,
        args.datapath,
        args.train_type,
        0.0, # randomness
        args.stack,
        args.num_episodes,
        False
    )
    
    data_variance = torch.var(observations / 255.0)
    
    print(f"Data variance: {data_variance}")
    print(f"Data shape: {observations.shape}")
        
    #shuffle and validation split
    indices = list(range(len(observations)))
    random.shuffle(indices)
    train_indices = indices[:int(len(observations) * args.val_split_ratio)]
    val_indices = indices[int(len(observations) * args.val_split_ratio):]

    observations, observations_val = observations[train_indices], observations[val_indices]
    print(f"Train size: {len(observations)} | Val size: {len(observations_val)}")
    train_data_loader = torch.utils.data.DataLoader(
        observations, batch_size=args.bs, shuffle=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        observations_val, batch_size=args.bs, shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    encoder = Encoder(
        args.stack,
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    )
    decoder = Decoder(
        args.stack,
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    )
    quantizer = VectorQuantizer(
        args.embedding_dim, args.num_embeddings, args.commitment_cost,
    )
    vqvae = VQVAEModel(encoder, decoder, quantizer).to(device)

    save_model_root = f'trained_models/vqvae_models/{args.env}'
    tensorboard_root = f'logs/vqvae/{args.env}'

    save_dir = f'seed_{args.seed}_stack_{args.stack}_ep_{args.num_episodes}_train_type_{args.train_type}'
    
    os.makedirs(f'{save_model_root}/{save_dir}', exist_ok=True)
        
    import datetime
    now_time = datetime.datetime.now()
    date_dir = str(now_time.strftime("%Y_%m_%d_%H_%M_%S"))
    
    writer = SummaryWriter(f'{tensorboard_root}/{date_dir}_{save_dir}')


    vqvae_optimizer = torch.optim.Adam(vqvae.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.n_epochs)):

        vqvae.train()
        
        train_loss = 0
        train_recon_error = 0
        train_vq_loss = 0
        count = 0
        for xx in train_data_loader:
            xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
            vqvae_optimizer.zero_grad()
            z, x_recon, vq_loss, quantized, _ = vqvae(xx)
            vq_loss = vq_loss.mean()
            recon_error = torch.mean((x_recon - xx) ** 2) / data_variance
            loss = recon_error + vq_loss
            loss.backward()

            vqvae_optimizer.step()
            
            train_loss += loss.item() * len(xx)
            train_recon_error += recon_error.item() * len(xx)
            train_vq_loss += vq_loss.item() * len(xx)
            count += len(xx)
            
            
        train_loss /= count
        train_recon_error /= count
        train_vq_loss /= count
        
        
        test_loss = 0
        test_recon_error = 0
        test_vq_loss = 0
        count = 0
                
        vqvae.eval()
        with torch.no_grad():
            for xx in test_data_loader:
                xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
                z, x_recon, vq_loss, quantized, _ = vqvae(xx)
                vq_loss = vq_loss.mean()
                recon_error = torch.mean((x_recon - xx) ** 2) / data_variance
                loss = recon_error + vq_loss
                
                test_loss += loss.item() * len(xx)
                test_recon_error += recon_error.item() * len(xx)
                test_vq_loss += vq_loss.item() * len(xx)
                count += len(xx)
                
        test_loss /= count
        test_recon_error /= count
        test_vq_loss /= count
        
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Recon Error/train', train_recon_error, epoch)
        writer.add_scalar('Recon Error/test', test_recon_error, epoch)
        writer.add_scalar('VQ Loss/train', train_vq_loss, epoch)
        writer.add_scalar('VQ Loss/test', test_vq_loss, epoch)
        print (f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Recon Error: {train_recon_error:.4f} | Test Recon Error: {test_recon_error:.4f} | Train VQ Loss: {train_vq_loss:.4f} | Test VQ Loss: {test_vq_loss:.4f}')
        
    
    torch.save(vqvae.state_dict(), f'{save_model_root}/{save_dir}/model.torch')   
    writer.close()

if __name__ == "__main__":
    args = get_args()
    # args.env = 'Amidar'
    # args.num_episodes = 2
    # args.n_epochs = 1
    train(args)