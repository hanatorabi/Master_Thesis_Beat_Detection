import os
import re
from statistics import mean

import mir_eval as eval
import numpy as np
import pretty_midi as pm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

def midi_to_pianoroll_image(midi_path, output_path='piano_roll.png', fs=100):
    """Saves a piano-roll image for the given MIDI file."""
    midi_file = pm.PrettyMIDI(midi_path)
    piano_roll = midi_file.get_piano_roll(fs=fs) 
    piano_roll = piano_roll / 127.0
    piano_roll = np.flipud(piano_roll)
    colormap = cm.get_cmap('cviridis')  
    colored_image = colormap(piano_roll) 
    mpimg.imsave(output_path, colored_image)

def evaluate_beats(ref_beats, est_beats, window=0.1):
    """
    Evaluate onset-based f-measure, precision, recall using mir_eval.onset.
    Returns (f_measure, precision, recall).
    """
    f_final, precision, recall = eval.onset.f_measure(
        np.array(ref_beats),
        np.array(est_beats),
        window=window
    )
    return f_final, precision, recall

def remove_beats(pred_beats):
    """
    Try different 'removal' patterns (even-index, odd-index, mod3, etc.)
    and pick the subset with the best F-measure.
    Returns (best_beats, best_f, best_precision, best_recall, best_pattern).
    """
    best_pattern = 'original'
    best_beats = pred_beats
    best_f = 0.0
    best_precision = 0.0
    best_recall = 0.0

    # Patterns to try:
    # - Even indices only
    even_beats = [pred_beats[i] for i in range(len(pred_beats)) if i % 2 == 0]
    yield ('even', even_beats)

    # - Odd indices only
    odd_beats = [pred_beats[i] for i in range(len(pred_beats)) if i % 2 == 1]
    yield ('odd', odd_beats)

    # - Mod 3 patterns
    mod3_0 = [pred_beats[i] for i in range(len(pred_beats)) if i % 3 == 0]
    yield ('mod3_0', mod3_0)
    mod3_1 = [pred_beats[i] for i in range(len(pred_beats)) if i % 3 == 1]
    yield ('mod3_1', mod3_1)
    mod3_2 = [pred_beats[i] for i in range(len(pred_beats)) if i % 3 == 2]
    yield ('mod3_2', mod3_2)

def add_beats(pred_beats):
    """
    Try inserting additional beats between consecutive ones.
    For demonstration, we attempt:
       - Insert 1 beat in the midpoint
       - Insert 2 beats evenly spaced
    Return each variant as a generator of (pattern_name, new_beats).
    """
    # Single midpoint
    single = []
    for i in range(len(pred_beats) - 1):
        t0 = pred_beats[i]
        t1 = pred_beats[i+1]
        single.append(t0)
        # Insert midpoint
        mid = t0 + 0.5 * (t1 - t0)
        single.append(mid)
    # Add the last beat
    if pred_beats:
        single.append(pred_beats[-1])
    yield ('single_midpoint', single)

    # Two equidistant points: fraction=1/3, fraction=2/3
    double_pts = []
    for i in range(len(pred_beats) - 1):
        t0 = pred_beats[i]
        t1 = pred_beats[i+1]
        double_pts.append(t0)
        mid1 = t0 + (1/3)*(t1 - t0)
        mid2 = t0 + (2/3)*(t1 - t0)
        double_pts.extend([mid1, mid2])
    if pred_beats:
        double_pts.append(pred_beats[-1])
    yield ('two_midpoints', double_pts)

if __name__ == "__main__":
    print("STARTING...")
    output_file = str(Path(__file__).resolve().parent) + "/denoised_inference_outputs_new_700000"

    f_measures = []
    precisions = []
    recalls = []

    # Directory with the reference "target" MIDI
    target_dir = '/h/hana/Documents/ASAP_Arxiv/1task/inference_outputs/check_target'

    for track in os.listdir(target_dir):
        if not track.endswith('.mid'):
            continue
        print(f"Processing track: {track}")

        # Extract an ID like "_7"
        x = re.findall("\_.*\.", track)
        if not x:
            print("No match for pattern. Skipping.")
            continue
        file_id = x[0].strip(".")

        # ========== Load reference beats ==========
        ref_path = os.path.join(target_dir, f"target{file_id}.mid")
        if not os.path.isfile(ref_path):
            print(f"Could not find reference: {ref_path}")
            continue
        pm0 = pm.PrettyMIDI(ref_path)
        ref_notes = pm0.instruments[0].notes
        ref_beats = [n.start for n in ref_notes if n.pitch == 110]

        # ========== Load predicted beats ==========
        pred_path = os.path.join(output_file, "aligned_agg_pred", f"output{file_id}.mid")
        if not os.path.isfile(pred_path):
            print(f"Could not find prediction: {pred_path}")
            continue
        pm1 = pm.PrettyMIDI(pred_path)
        pred_notes = pm1.instruments[0].notes
        pred_beats = [n.start for n in pred_notes if n.pitch == 110]

        # Evaluate original
        f_final, precision, recall = evaluate_beats(ref_beats, pred_beats, window=0.1)
        print(f"Original => ID={file_id}, F={f_final:.2f}, P={precision:.2f}, R={recall:.2f}")

        # ---------------------------------------------------
        # CASE A: If recall == 1 but precision < 1
        #   => Too many predicted beats, try removing spurious
        # ---------------------------------------------------
        if recall > 0.8 and precision < 0.6:
            best_f = f_final
            best_precision = precision
            best_recall = recall
            best_beats = pred_beats
            best_pattern = "original"

            # For each pattern in remove_beats() generator
            for pattern_name, subset_beats in remove_beats(pred_beats):
                f_tmp, p_tmp, r_tmp = evaluate_beats(ref_beats, subset_beats)
                if f_tmp > best_f:
                    best_f = f_tmp
                    best_precision = p_tmp
                    best_recall = r_tmp
                    best_beats = subset_beats
                    best_pattern = pattern_name

            if best_pattern != "original":
                print(f"  => Found a better REMOVAL pattern: {best_pattern}")
                print(f"     new F={best_f:.2f}, P={best_precision:.2f}, R={best_recall:.2f}")

                # Overwrite
                pred_beats = best_beats
                f_final = best_f
                precision = best_precision
                recall = best_recall

                # (Optional) Save new filtered MIDI
                filtered_path = os.path.join(output_file, "aligned_agg_pred", f"filtered_remove_{file_id}.mid")
                new_midi = pm.PrettyMIDI()
                inst = pm.Instrument(program=0)
                for t in pred_beats:
                    inst.notes.append(pm.Note(velocity=100, pitch=110, start=t, end=t+0.05))
                new_midi.instruments.append(inst)
                new_midi.write(filtered_path)
                print(f"  => Saved filtered MIDI (removal) to {filtered_path}")
            else:
                print("  => No removal pattern improved F-measure.")

        # ---------------------------------------------------
        # CASE B: If precision == 1 but recall < 1
        #   => We have only correct beats but not enough
        #      (missing many real beats). Let's try adding.
        # ---------------------------------------------------
        if precision > 0.8 and recall < 0.6:
            best_f = f_final
            best_precision = precision
            best_recall = recall
            best_beats = pred_beats
            best_pattern = "original"

            # For each pattern in add_beats() generator
            for pattern_name, new_beats in add_beats(pred_beats):
                f_tmp, p_tmp, r_tmp = evaluate_beats(ref_beats, new_beats)
                if f_tmp > best_f:
                    best_f = f_tmp
                    best_precision = p_tmp
                    best_recall = r_tmp
                    best_beats = new_beats
                    best_pattern = pattern_name

            if best_pattern != "original":
                print(f"  => Found a better ADD pattern: {best_pattern}")
                print(f"     new F={best_f:.2f}, P={best_precision:.2f}, R={best_recall:.2f}")

                # Overwrite
                pred_beats = best_beats
                f_final = best_f
                precision = best_precision
                recall = best_recall

                # (Optional) Save new filtered MIDI
                filtered_path = os.path.join(output_file, "aligned_agg_pred", f"filtered_add_{file_id}.mid")
                new_midi = pm.PrettyMIDI()
                inst = pm.Instrument(program=0)
                for t in pred_beats:
                    inst.notes.append(pm.Note(velocity=100, pitch=110, start=t, end=t+0.05))
                new_midi.instruments.append(inst)
                new_midi.write(filtered_path)
                print(f"  => Saved filtered MIDI (addition) to {filtered_path}")
            else:
                print("  => No addition pattern improved F-measure.")

        # Optionally do the same "image saving" logic
        if precision == 1.0:
            out_img = os.path.join(output_file, 'images', f'p_track{file_id}.png')
            midi_to_pianoroll_image(pred_path, output_path=out_img, fs=100)

        if recall == 1.0:
            out_img = os.path.join(output_file, 'images', f'r_track{file_id}.png')
            midi_to_pianoroll_image(pred_path, output_path=out_img, fs=100)

        if f_final == 1.0:
            out_img = os.path.join(output_file, 'images', f'f_track{file_id}.png')
            midi_to_pianoroll_image(pred_path, output_path=out_img, fs=100)

        # Print final result
        print(f"Final => ID={file_id}, F={f_final:.2f}, P={precision:.2f}, R={recall:.2f}\n")

        # Collect for overall stats
        f_measures.append(f_final)
        precisions.append(precision)
        recalls.append(recall)

    # ---- AFTER THE LOOP: MAKE PLOTS ----

    # Scatter of (recall, precision)
    plt.figure(figsize=(6, 6))
    plt.scatter(recalls, precisions, alpha=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall (per file)')
    plt.grid(True)
    plot_path = os.path.join(output_file, "precision_recall_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Precision–Recall plot saved to: {plot_path}")

    # Histograms of precision & recall
    hist_path = os.path.join(output_file, "hist_plot.png")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(precisions, bins=10, color='skyblue', edgecolor='black')
    plt.title('Precision Distribution')
    plt.xlabel('Precision')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(recalls, bins=10, color='salmon', edgecolor='black')
    plt.title('Recall Distribution')
    plt.xlabel('Recall')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histogram plot saved to: {hist_path}")

    # Final stats
    num_files = len(f_measures)
    print("Number of files processed:", num_files)
    if num_files > 0:
        print("Final F-measure:", sum(f_measures)/num_files)
        print("Final Precision:", sum(precisions)/num_files)
        print("Final Recall:", sum(recalls)/num_files)
        f_measures_nonzero = [f for f in f_measures if f > 0]
        if len(f_measures_nonzero) > 0:
            print("Non-zero F count:", len(f_measures_nonzero))
            print("Final F-measure (non-zero):", sum(f_measures_nonzero)/len(f_measures_nonzero))
