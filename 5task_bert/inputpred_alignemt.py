import pretty_midi as pm
import mir_eval as eval
import numpy as np
import os
from statistics import mean
import re
import torch
from pathlib import Path

output_file = str(
    Path(__file__).resolve().parent) + "/denoised_inference_outputs_new_" + str(700000)
for r in range(1, 11):
    os.mkdir(output_file + "/aligned_pred" + str(r))
    for track in os.listdir(output_file + '/pred' + str(r)):

        print(track, flush=True)
        x = re.findall("\_.*\.", track)

        pm0 = pm.PrettyMIDI(
            midi_file=
            '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/input/input'
            + str(x[0]) + 'mid')
        input_notes = pm0.instruments[0].notes
        pm1 = pm.PrettyMIDI(midi_file=output_file + '/pred' + str(r) +
                            '/output' + str(x[0]) + 'mid')
        pred_notes = pm1.instruments[0].notes
        init_tempo = pm0.get_tempo_changes()[1][0]

        midi_data = pm.PrettyMIDI(resolution=1024, initial_tempo=init_tempo)
        piano = pm.Instrument(
            program=pm.instrument_name_to_program('Acoustic Grand Piano'))

        for i in reversed(range(len(pred_notes))):
            if pred_notes[i].pitch != 110:
                pred_last = pred_notes[i]
                break

        for i in reversed(range(len(input_notes))):
            if input_notes[i].pitch == pred_last.pitch:
                input_start = input_notes[i].start
                break

        #shifting with beats
        if abs(input_start - pred_last.start) > 0.0001:

            dif = pred_last.start - input_start
            print(track, dif)
            for note in pred_notes:
                if note.start - dif < 0:
                    continue
                else:
                    note.start = note.start - dif
                    note.end = note.end - dif
                    piano.notes.append(note)
        else:
            piano.notes = pred_notes

        midi_data.instruments.append(piano)
        midi_data.write(output_file + '/aligned_pred' + str(r) + '/' + track)

print('done')
