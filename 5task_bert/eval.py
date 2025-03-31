import os
import re
from statistics import mean

import mir_eval as eval
import numpy as np
import pretty_midi as pm
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import shutil
# from sklearn.metrics import roc_curve, auc




def midi_to_pianoroll_image(midi_path, output_path='piano_roll.png', fs=100):
    midi_file = pm.PrettyMIDI(midi_path)
    piano_roll = midi_file.get_piano_roll(fs=fs) 
    piano_roll = piano_roll / 127.0
    piano_roll = np.flipud(piano_roll)
    colormap = cm.get_cmap('viridis')  
    colored_image = colormap(piano_roll) 

    mpimg.imsave(output_path, colored_image)

# destination_folder = Path("/h/hana/Documents/NextToken/Nocturne2023_beat/5task/inference_outputs_new_700000", 
#                           "recall == 0.5 and 0.2 < precision < 0.6")
# destination_folder.mkdir(parents=True, exist_ok=True)



num_iter = 10
threshold = 5
print('Threshold is : ', threshold)
output_file = str(
    Path(__file__).resolve().parent) + "/inference_outputs_new_" + str(700000)
f_measures = []
precisions = []
recalls = []
for track in os.listdir(
        '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target'):
    # print(track)
    x = re.findall("\_.*\.", track)

    midi_files = []
    midi_notes = []

    pm0 = pm.PrettyMIDI(
        midi_file=
        '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target/target'
        + str(x[0]) + 'mid')
    notes = pm0.instruments[0].notes
    midi_notes.append(notes)
    midi_files.append(pm0)

    beat_array = []
    temp = []
    for note in notes:
        if note.pitch == 110:
            temp.append(note.start)
    beat_array.append(temp)

    pm1 = pm.PrettyMIDI(
        midi_file=
        output_file +'/aligned_agg_pred'+'/output'
        + str(x[0]) + 'mid')
    notes_pred = pm1.instruments[0].notes
    pred_beat = []
    for note in notes_pred:
        if note.pitch == 110:
            pred_beat.append(note.start)

    # f_final = eval.beat.f_measure(np.array(beat_array[0]),np.array(pred_beat),f_measure_threshold = 0.1)
    
    f_final, precision, recall = eval.onset.f_measure(np.array(beat_array[0]),
                                                      np.array(pred_beat),
                                               window=0.1)
    
    # fpr, tpr, thresholds = roc_curve(np.array(beat_array[0]), np.array(pred_beat))
    # roc_auc = auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2,
    #         label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate (Recall)')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.grid(True)

    # # Save the ROC curve as an image file (e.g., PNG)
    # plt.savefig("roc_curve.png", dpi=300)
    # plt.close()  # Close the figure to free memory

    # print("ROC curve saved as 'roc_curve.png'")


    # if recall == 0.5 and 0.2 < precision < 0.6:
    #     print( "pattern" ,str(x[0]), recall, precision, f_final)
    #     shutil.copy2('/h/hana/Documents/NextToken/Nocturne2023_beat/5task/inference_outputs_new_700000/aligned_agg_pred/output'
    #     + str(x[0]) + 'mid',str(destination_folder)+ '/output'
    #     + str(x[0]) + 'mid')
    #     shutil.copy2('/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target/target'
    #     + str(x[0]) + 'mid',str(destination_folder)+ '/target'
    #     + str(x[0]) + 'mid')


    # if precision == 1 :
    #     # print( str(x[0]), recall, precision, f_final)
    #     midi_to_pianoroll_image(output_file +'/aligned_agg_pred/output'
    #     + str(x[0]) + 'mid', output_path=output_file +'/images/p_track'
    #     + str(x[0]) + 'png', fs=100)
    #     midi_to_pianoroll_image('/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target/target'
    #     + str(x[0]) + 'mid', output_path=output_file +'/images/track'
    #     + str(x[0]) + 'png', fs=100)

    # if recall == 1:
    #     # print( str(x[0]), recall, precision, f_final)
    #     midi_to_pianoroll_image(output_file +'/aligned_agg_pred/output'
    #     + str(x[0]) + 'mid', output_path=output_file +'/images/r_track'
    #     + str(x[0]) + 'png', fs=100)
    #     midi_to_pianoroll_image('/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target/target'
    #     + str(x[0]) + 'mid', output_path=output_file +'/images/track'
    #     + str(x[0]) + 'png', fs=100)

    # if f_final == 1 :
    #     # print( str(x[0]), recall, precision)
    #     midi_to_pianoroll_image(output_file +'/aligned_agg_pred/output'
    #     + str(x[0]) + 'mid', output_path=output_file +'/images/f_track'
    #     + str(x[0]) + 'png', fs=100)

    # midi_to_pianoroll_image('/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/aligned_agg_pred/output'
    # + str(x[0]) + 'mid', output_path=output_file +'/test_images/track'
    # + str(x[0]) + 'png', fs=100)

    print('Last Prediction: ', f_final)

    f_measures.append(f_final)
    precisions.append(precision)
    recalls.append(recall)


plt.figure(figsize=(6, 6))
plt.scatter(recalls, precisions, alpha=0.7)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall (per file)')
plt.grid(True)
plt.legend()

plot_path = os.path.join(output_file, "precision_recall_plot.png")
plt.savefig(plot_path)
plt.close()
print(f"Precision–Recall plot saved to: {plot_path}")








# hist_path = os.path.join(output_file, "hist_plot.png")
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
# plt.hist(precisions, bins=10, color='skyblue', edgecolor='black')
# plt.title('Precision Distribution')
# plt.xlabel('Precision')
# plt.ylabel('Count')

# # 2) Recall histogram
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
# plt.hist(recalls, bins=10, color='salmon', edgecolor='black')
# plt.title('Recall Distribution')
# plt.xlabel('Recall')
# plt.ylabel('Count')

# # Ensure everything fits without overlapping
# plt.tight_layout()

# # --- Save the figure as an image ---
# plt.savefig(hist_path, dpi=300)
# plt.close()



print(len(f_measures))
print('Final F_Measure: ', sum(f_measures) / len(f_measures))
print('Final Percision: ', sum(precisions) / len(precisions))
print('Final Recall: ', sum(recalls) / len(recalls))
f_measures_new = [i for i in f_measures if i > 0]
print(len(f_measures_new))
print('Final F_Measure: ', sum(f_measures_new) / len(f_measures_new))

# print(f_measures)
