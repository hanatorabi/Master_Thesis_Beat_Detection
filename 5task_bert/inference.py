import torch
from midi_processing import dat2mid_anna, mid2dat_anna
from model import *
from data import MidiToken
from data import *
import os
import re
import pretty_midi as pm
import mir_eval as eval
import numpy as np
from statistics import mean
from argparse import Namespace
from pathlib import Path

MASK_IDX = 390
PAD_IDX = 391
device = "cuda:0"

ckpt_file = '/scratch/ssd004/scratch/hana/bert_5task/bert_5task1000000.pt'
epoch = re.search(r'task(\d+)', Path(ckpt_file).stem).group(1)
ckpt = torch.load(ckpt_file)

args_dict = {
    "pretrained": 1,
    "layers": 6,
    "embed_dim": 512,
    "vocab_size": 897,
    "pad_token_id": PAD_IDX
}
args = Namespace(**args_dict)
model = BertModel(args)
model.load_state_dict(ckpt['model_state_dict'])
model.cuda()
model.eval()
print("model loading done")

output_file = str(
    Path(__file__).resolve().parent) + "/inference_outputs_new_" + str(epoch)
if not os.path.isdir(output_file):
    os.mkdir(output_file)
with open(output_file + "/model_checkpoint_path.txt", 'w') as file:
    file.write(ckpt_file)

######for regenrating and checking if any prediction is empty
with torch.no_grad():
    for preds in range(9, 11):
        os.mkdir(output_file + "/pred" + str(preds))
        for n in range(20):
            for i in range(20):

                if n < 21 and i < 21:
                    print("input_" + str(n) + "_" + str(i + 1) + ".mid",
                          flush=True)
                    pm0 = pm.PrettyMIDI(
                        midi_file=
                        "/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/input/input_"
                        + str(n) + "_" + str(i + 1) + ".mid")
                    init_tempo = pm0.get_tempo_changes()[1][0]

                    inp = torch.full((1, 512), PAD_IDX).long()
                    mid2dat_input = mid2dat_anna(
                        "/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/input/input_"
                        + str(n) + "_" + str(i + 1) + ".mid")[:500]
                    task_token = torch.tensor([[3]]).cuda()
                    inp[0, :len(mid2dat_input)] = torch.tensor(
                        list(map(MidiToken.key_mapping, mid2dat_input)))
                    # inp = torch.cat((task_token, inp[:, :-1]), 1)
                    inp = inp.to(device)

                    for t in range(1):
                        start = torch.tensor([[388]],
                                             dtype=torch.long).to(device)
                        pred_tensor = inp[:, 1:6].clone().detach()
                        pred_tensor = torch.cat((start, pred_tensor[:, :-1]),
                                                1)
                        pred_tensor = torch.cat(
                            (task_token, pred_tensor[:, :-1]), 1)
                        pred_tensor = pred_tensor.cuda()

                        ### Teacher-forcing
                        j = 6
                        time = 0
                        curr_vel = 0
                        while pred_tensor.size(1) < 512:
                            output = model(inp, pred_tensor)[0, -1, :]
                            pred = torch.multinomial(
                                torch.nn.functional.softmax(output, dim=0),
                                num_samples=1)
                            pred_token = MidiToken.tok_mapping(pred)
                            if pred_token is None:
                                while pred_token is None:
                                    pred = torch.multinomial(
                                        torch.nn.functional.softmax(output,
                                                                    dim=0),
                                        num_samples=1)
                                    pred_token = MidiToken.tok_mapping(pred)

                            inp_token = MidiToken.tok_mapping(inp[0, j])

                            if pred_token.type == "STOP":
                                break

                            if inp_token.type == "SET_VELOCITY":
                                curr_vel = inp_token.value

                            if pred_token.type == "SET_VELOCITY":
                                temp_tensor = torch.cat(
                                    (pred_tensor, pred.unsqueeze(0)), dim=1)
                                output = model(inp, temp_tensor)[0, -1, :]
                                pred = torch.multinomial(
                                    torch.nn.functional.softmax(output, dim=0),
                                    num_samples=1)
                                pred_token = MidiToken.tok_mapping(pred)
                                if pred_token.type == "BEAT":
                                    pred_tensor = torch.cat(
                                        (temp_tensor, pred.unsqueeze(0)),
                                        dim=1)
                                    temp_token = MidiToken.key_mapping(
                                        MidiToken("SET_VELOCITY", curr_vel, 0))
                                    pred_tensor = torch.cat(
                                        (pred_tensor, torch.tensor([
                                            temp_token
                                        ]).unsqueeze(0).to(device)),
                                        dim=1)
                                    continue
                                else:
                                    pred_tensor = torch.cat(
                                        (pred_tensor, inp[:, j].unsqueeze(0)),
                                        dim=1)
                                    j += 1

                            elif pred_token.type == "BEAT":
                                pred_tensor = torch.cat(
                                    (pred_tensor, pred.unsqueeze(0)), dim=1)
                                continue

                            elif inp_token.type == "TIME_SHIFT" and pred_token.type == "TIME_SHIFT" and pred_token.value <= inp_token.value:
                                time = inp_token.value
                                true_time_rem = time
                                temp_tensor = pred_tensor.clone()

                                while time > 0:
                                    if pred_token.type == "TIME_SHIFT" and pred_token.value <= time:
                                        temp_tensor = torch.cat(
                                            (temp_tensor, pred.unsqueeze(0)),
                                            dim=1)
                                        time -= pred_token.value
                                    elif pred_token.type == "SET_VELOCITY":
                                        temp_tensor = torch.cat(
                                            (temp_tensor, pred.unsqueeze(0)),
                                            dim=1)
                                    elif pred_token.type == "BEAT":
                                        temp_tensor = torch.cat(
                                            (temp_tensor, pred.unsqueeze(0)),
                                            dim=1)
                                        pred_tensor = temp_tensor.clone()
                                        true_time_rem = time
                                    else:
                                        break

                                    output = model(inp, temp_tensor)[0, -1, :]
                                    pred = torch.multinomial(
                                        torch.nn.functional.softmax(output,
                                                                    dim=0),
                                        num_samples=1)
                                    pred_token = MidiToken.tok_mapping(pred)

                                if true_time_rem > 0:
                                    temp_token = MidiToken.key_mapping(
                                        MidiToken("TIME_SHIFT", true_time_rem,
                                                  0))
                                    pred_tensor = torch.cat(
                                        (pred_tensor, torch.tensor([
                                            temp_token
                                        ]).unsqueeze(0).to(device)),
                                        dim=1)
                                j += 1

                            else:
                                pred_tensor = torch.cat(
                                    (pred_tensor, inp[:, j].unsqueeze(0)),
                                    dim=1)
                                j += 1

                        pred_seq = list(
                            map(MidiToken.tok_mapping, pred_tensor[0].cpu()))
                        dat2mid_anna(pred_seq,
                                     init_tempo,
                                     fname=output_file + "/pred" + str(preds) +
                                     "/output_" + str(n) + "_" + str(i + 1) +
                                     ".mid")