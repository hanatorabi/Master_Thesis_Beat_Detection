import torch
from midi_processing import dat2mid_anna, mid2dat_anna
from data import MidiToken
from data import *
import os
import re
import pretty_midi as pm
import mir_eval as eval
import numpy as np
from statistics import mean
from pathlib import Path

MASK_IDX = 390
PAD_IDX = 391
device = "cuda:0"


output_file = str(
    Path(__file__).resolve().parent) + "/inference_outputs_new_" + str(700000)

# # Finding agg_pred
shift_size = 0.01 #10 ms
num_iter = 10
threshold = 4
print('Threshold is : ', threshold)
f_measures = []
os.mkdir(output_file + '/aligned_agg_pred')
for track in os.listdir('/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/aligned_target'):
    print(track)
    x = re.findall("\_.*\.", track)
    

    midi_files = []
    midi_notes = []

    pm0 = pm.PrettyMIDI(midi_file= '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/aligned_target/target'+str(x[0])+'mid')
    target_notes = pm0.instruments[0].notes
    pm3 = pm.PrettyMIDI(midi_file= '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/input/input'+str(x[0])+'mid')
    input_notes = pm3.instruments[0].notes
    init_tempo = pm3.get_tempo_changes()[1][0]
    midi_notes.append(target_notes)
    midi_files.append(pm0)

    beat_array = []
    temp =[]
    for note in target_notes:
        if note.pitch == 110:
            temp.append(note.start)
    beat_array.append(temp)
    # print(beat_array)
    for n in range(num_iter):
        pm1 = pm.PrettyMIDI(midi_file=output_file + '/aligned_pred'+str(n+1)+'/output'+str(x[0])+'mid')
        notes = pm1.instruments[0].notes
        midi_notes.append(notes)
        temp = []
        for note in notes:
            if note.pitch == 110:
                temp.append(note.start)
        beat_array.append(temp)




    end_of_seq = target_notes[-1].end
    n_bins = (end_of_seq // shift_size) + 1
    n_bins = n_bins.astype(int)        
    one_hot_tensor = torch.zeros(num_iter + 1, n_bins)
   
    for j in range(n_bins):
        first_period = j * shift_size
        end_period = (j+1) * shift_size
        for n in range(1,num_iter+1):
            for l in beat_array[n]:
                if l >= first_period and l < end_period:
                    one_hot_tensor[n-1,j] = l

    window_len = 25
    
    i = 0
    while i < n_bins:
        count = 0
        sum_beats = 0
        for j in range(num_iter):
            if any(one_hot_tensor[j,i:i+window_len]) > 0:
                count += 1
                sum_beats += torch.sum(one_hot_tensor[j,i:i+window_len]) 
        if count >= threshold: 
            one_hot_tensor[-1, i] = sum_beats / count
            i += window_len
        else:
            i += 1





    midi_data = pm.PrettyMIDI(resolution=1024, initial_tempo=init_tempo)
    piano = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
    final_pred_start = one_hot_tensor[-1,:].tolist()
    final_pred_start = [i for i in final_pred_start if i != 0]
    print(final_pred_start)

    piano.notes = input_notes 
    for t in final_pred_start:
        piano.notes.append(pm.Note(velocity=127, pitch=110, start=t, end=t + 0.2))
    (piano.notes).sort(key = lambda x: x.start)

    for note in piano.notes:
        if note.end > (target_notes[-1].end + 0.5):
            (piano.notes).remove(note)
            
    midi_data.instruments.append(piano)
    midi_data.write(output_file+'/aligned_agg_pred/output'+str(x[0])+'mid')



print('done')



























































