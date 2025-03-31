import os
import time
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import *
from midi_processing import *
from model import *
print('start', flush=True)
print('data loading moved to gpu', flush=True)

# initialize data

train = MidiDataset(filename1= '/h/hana/Documents/ASAP_Arxiv/dataset/ASAP_valid.pickle',filename2='/h/hana/Documents/ISMIR submission Beat Detection/with pedal/Adding MAESTRO/dataset/maestro_valid.pickle',train=True,seq_len=512, train_split = 1)
valid = MidiDataset(filename1 = '/h/hana/Documents/ASAP_Arxiv/dataset/ASAP_valid.pickle',filename2='/h/hana/Documents/ISMIR submission Beat Detection/with pedal/Adding MAESTRO/dataset/maestro_valid.pickle',train=True,seq_len=512, train_split = 1)



# initialize model, loss, optimizer
PAD_IDX = 391
args_dict = {
    "pretrained": 1,
    "layers": 6,
    "embed_dim": 512,
    "vocab_size": 897,
    "pad_token_id": PAD_IDX
}
args = Namespace(**args_dict)
model = BertModel(args)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

batch_size = 20
print_every = 50
checkpoint_path = '/scratch/ssd004/scratch/hana/bert_5task/bert_5task'
checkpoint_backup = '/scratch/ssd004/scratch/hana/bert_5task/backup_bert_5task'

# load model if exists
if not os.path.isfile(checkpoint_backup+'.pt'):
    print('backup')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time': time.time(),
            'train_losses': [],
            'valid_losses': [],
            'valid_tasks': [],
            'step': 0,
        }, checkpoint_backup + '.pt')

if not os.path.isfile(checkpoint_path+'.pt'):
    print('start')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time': time.time(),
            'train_losses': [],
            'valid_losses': [],
            'valid_tasks': [],
            'step': 0,
        }, checkpoint_path + '.pt')

# load last checkpoint to continue
print('continue')
checkpoint = torch.load(checkpoint_path + '.pt',
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
all_train_losses = checkpoint['train_losses']
all_valid_losses = checkpoint['valid_losses']
all_valid_tasks = checkpoint['valid_tasks']
step = checkpoint['step']
print("Num trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad),
      flush=True)

# training loop
while True:
    model.train()
    batch = train.get_batch(batch_size)
    inp = batch["inp"].cuda()
    tgt = batch["trg"].cuda()

    src_pad_mask = (inp == PAD_IDX).cuda()
    tgt_pad_mask = (tgt == PAD_IDX).cuda()

    optimizer.zero_grad()
    out = model(inp, tgt)
    loss = criterion(out[:, :-1][~tgt_pad_mask[:, 1:]],
                     tgt[:, 1:][~tgt_pad_mask[:, 1:]])
    loss.backward()
    optimizer.step()

    step += 1

    # Print progress/Validation
    if step % print_every == 0:
        model.eval()
        with torch.no_grad():
            vbatch = valid.get_batch(batch_size)
            inp = vbatch["inp"].cuda()
            tgt = vbatch["trg"].cuda()

            src_pad_mask = (inp == PAD_IDX).cuda()
            tgt_pad_mask = (tgt == PAD_IDX).cuda()

            out = model(inp, tgt)
            valid_loss = criterion(out[:, :-1][~tgt_pad_mask[:, 1:]],
                                   tgt[:, 1:][~tgt_pad_mask[:, 1:]])

            valid_loss1 = criterion(out[0:4, :-1][~tgt_pad_mask[0:4, 1:]],
                                    tgt[0:4, 1:][~tgt_pad_mask[0:4, 1:]])
            valid_loss2 = criterion(out[4:8, :-1][~tgt_pad_mask[4:8, 1:]],
                                    tgt[4:8, 1:][~tgt_pad_mask[4:8, 1:]])
            valid_loss3 = criterion(out[8:12, :-1][~tgt_pad_mask[8:12, 1:]],
                                    tgt[8:12, 1:][~tgt_pad_mask[8:12, 1:]])
            valid_loss4 = criterion(out[12:16, :-1][~tgt_pad_mask[12:16, 1:]],
                                    tgt[12:16, 1:][~tgt_pad_mask[12:16, 1:]])
            valid_loss5 = criterion(out[16:20, :-1][~tgt_pad_mask[16:20, 1:]],
                                    tgt[16:20, 1:][~tgt_pad_mask[16:20, 1:]])
        print(step,
              "Train Loss: ",
              loss.detach(),
              "Validation Loss: ",
              valid_loss.detach(),
              "Validation Lossses for each task : ",
              valid_loss1.detach(),
              valid_loss2.detach(),
              valid_loss3.detach(),
              valid_loss4.detach(),
              valid_loss5.detach(),
              flush=True)
        all_train_losses.append(loss.detach())
        all_valid_losses.append(valid_loss.detach())
        all_valid_tasks.append([ valid_loss1.detach(),
              valid_loss2.detach(),
              valid_loss3.detach(),
              valid_loss4.detach(),
              valid_loss5.detach(),])
        print("saving model", flush=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'time': time.time(),
                'train_losses': all_train_losses,
                'valid_losses': all_valid_losses,
                'valid_tasks': all_valid_tasks,
                'step': step,
            }, checkpoint_path + '.pt')

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'time': time.time(),
                'train_losses': all_train_losses,
                'valid_losses': all_valid_losses,
                'valid_tasks': all_valid_tasks,
                'step': step,
            }, checkpoint_backup + '.pt')
        print("saving done!", flush=True)

        if step in [
                50, 80000, 100000, 200000, 300000, 320000, 400000, 500000,
                600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000
        ]:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'time': time.time(),
                    'train_losses': all_train_losses,
                    'valid_losses': all_valid_losses,
                    'step': step,
                }, checkpoint_backup + str(step) + '.pt')

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'time': time.time(),
                    'train_losses': all_train_losses,
                    'valid_losses': all_valid_losses,
                    'step': step,
                }, checkpoint_path + str(step) + '.pt')

            print("saving done!", flush=True)
