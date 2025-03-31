import os
import pickle
import pandas as pd
import pretty_midi as pm
from pathlib import Path
from data import MidiToken




def dat2mid_anna(seq, tempo, fname="test.mid"):
    '''
    Given a sequence of MIDI events, write a MIDI file.
        Parameters:
            seq (list of MidiToken): Sequence of MIDI event objects
            fname (str): Output filename
        Returns:
            None
    '''
    assert seq is not None
    assert isinstance(seq[0], MidiToken)
    start_times = [-1] * 128 # -1=inactive, else val=start_time
    velocities  = [-1] * 128 # -1=inactive, else val=velocity
    curr_time = 0.0
    curr_vel = 0
    midi_data = pm.PrettyMIDI(resolution=1024, initial_tempo=tempo)
    piano = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
    for event in seq:
        if event.type == "NOTE_ON" and event.value != 110:
            start_times[event.value] = curr_time
            velocities[event.value] = curr_vel
        elif event.type == "BEAT":
          note = pm.Note(velocity=curr_vel, pitch=110, start=curr_time, end=curr_time + 0.2)
          piano.notes.append(note)
        elif event.type == "NOTE_OFF" and start_times[event.value] != -1:
            note = pm.Note(velocity=velocities[event.value], pitch=event.value, start=start_times[event.value], end=curr_time)
            piano.notes.append(note)
            start_times[event.value] = -1
            velocities[event.value] = -1
        elif event.type == "TIME_SHIFT":
            curr_time += event.value / 1000
        elif event.type == "SET_VELOCITY":
            curr_vel = event.value
        elif event.type == "PEDAL_ON":
            piano.control_changes.append(pm.ControlChange(64, 127, curr_time))
        elif event.type == "PEDAL_OFF":
            piano.control_changes.append(pm.ControlChange(64, 0, curr_time))
    midi_data.instruments.append(piano)
    # for i in midi_data.instruments[0].control_changes:
    #     print(i)
    midi_data.write(fname)

def mid2dat_anna(midi_path):
    '''
    Given a MIDI file, convert into a sequence of MIDI events.
        Parameters:
            midi_path (str/Path): Input MIDI filename
        Returns:
            arr (list): List of MidiToken event objects
    '''
    arr = []
    if not isinstance(midi_path, str):
        midi_path = midi_path.as_posix()
    midi_data = pm.PrettyMIDI(midi_path, resolution=1024)
    pedal_events = [c for c in midi_data.instruments[0].control_changes if c.number == 64]
    # for i in pedal_events: print(i)
    x = midi_data.instruments[0].get_piano_roll(fs=100, pedal_threshold=None) # shape=(pitch, timestep)

    active_notes = []   # unended NOTE_ON pitches
    time_acc = -10      # track time passed (ms) since last TIME_SHIFT (start at -10 to offset first increment)
    curr_vel = 0        # last SET_VELOCITY value
    pedal = 0           # status of pedal: 0==off, 1==on
    # Iterate over timesteps
    for t in range(x.shape[1]):
        time_acc += 10
        for p in range(x.shape[0]):
            # Check for pedal events
            if pedal_events and t/100 >= pedal_events[0].time:
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
                    time_acc = 0
                if pedal == 0 and pedal_events[0].value >= 64:
                    arr.append(MidiToken("PEDAL_ON", 0, t/100))
                    pedal = 1
                elif pedal == 1 and pedal_events[0].value < 64:
                    arr.append(MidiToken("PEDAL_OFF", 0, t/100))
                    pedal = 0
                del pedal_events[0]
            # When velocity is not 0
            if x[p,t] and p not in active_notes:
                active_notes.append(p)
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
                    time_acc = 0
                if (x[p,t]//4)*4 != curr_vel:
                    curr_vel = int(x[p,t]//4)*4
                    arr.append(MidiToken("SET_VELOCITY", curr_vel, t/100))
                if p == 110:
                    arr.append(MidiToken("BEAT", p, t/100))
                else:
                    arr.append(MidiToken("NOTE_ON", p, t/100))
            # When a note ends
            elif not x[p,t] and p in active_notes:
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
                    time_acc = 0
                active_notes.remove(p)
                arr.append(MidiToken("NOTE_OFF", p, t/100))
        if time_acc == 1000:
            arr.append(MidiToken("TIME_SHIFT", 1000, t/100))
            time_acc = 0
    # Write final NOTE_OFFs
    if active_notes:
        time_acc += 10
        arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
        for p in active_notes:
            if p != -1:
                arr.append(MidiToken("NOTE_OFF", p, t/100))
    return arr


    





def beat2dat_anna(midi_path):
    '''
    Given a MIDI file, convert into a sequence of MIDI events.
        Parameters:
            midi_path (str/Path): Input MIDI filename
        
        Returns:
            arr (list): List of MidiToken event objects
    '''
    arr = []
    if not isinstance(midi_path, str):
        midi_path = midi_path.as_posix()
    midi_data = pm.PrettyMIDI(midi_path,resolution=1024)
    x = midi_data.instruments[0].get_piano_roll(fs=100) # shape=(pitch, timestep)
    
    active_notes = [] # unended NOTE_ON pitches
    time_acc = -10 # track time passed (ms) since last TIME_SHIFT (start at -10 to offset first increment)
    curr_vel = 127 # last SET_VELOCITY value
    
    # Iterate over timesteps
    for t in range(x.shape[1]):
        
        time_acc += 10
        for p in range(x.shape[0]):
            # When a note starts
            if x[p,t] and p not in active_notes:
                active_notes.append(p)
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
                    time_acc = 0
                if (x[p,t]//4)*4 != curr_vel:
                    curr_vel = (x[p,t]//4)*4
                    # arr.append(MidiToken("SET_VELOCITY", curr_vel, t/100))
                arr.append(MidiToken("BEAT", 110, t/100))
            # When a note ends
            elif not x[p,t] and p in active_notes:
                # if time_acc:
                #     arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
                #     time_acc = 0
                active_notes.remove(p)
                # arr.append(MidiToken("NOTE_OFF", p, t/100))
        if time_acc == 1000:
            arr.append(MidiToken("TIME_SHIFT", 1000, t/100))
            time_acc = 0
    # Write final NOTE_OFFs
    if active_notes:
        time_acc += 10
        if time_acc > 1000:
          arr.append(MidiToken("TIME_SHIFT", time_acc, t/100))
        # for p in active_notes:
        #     if p != -1:
        #         # arr.append(MidiToken("NOTE_OFF", p, t/100))
    return arr

    


def dat2beat_anna(seq,tempo ,fname="test.mid"):
    '''
    Given a sequence of MIDI events, write a MIDI file.
        Parameters:
            seq (list of MidiToken): Sequence of MIDI event objects
            fname (str): Output filename
        
        Returns:
            None
    '''
    assert seq is not None
    assert isinstance(seq[0], MidiToken)
    curr_notes = [-1] * 128 # -1=inactive, else val=start_time
    curr_time = 0.0
    curr_vel = 127
    midi_data = pm.PrettyMIDI(resolution=1024, initial_tempo=tempo)
    piano = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
    for event in seq:
        if event.type == "BEAT":# and curr_notes[event.value] != -1:
            curr_notes[event.value] = curr_time
            note = pm.Note(velocity=curr_vel, pitch=110, start=curr_notes[event.value], end=curr_time + 0.2)
            piano.notes.append(note)
            curr_notes[event.value] = -1
        # if event.type == "NOTE_OFF" and curr_notes[event.value] != -1:
        #     note = pm.Note(velocity=curr_vel, pitch=event.value, start=curr_notes[event.value], end=curr_time)
        #     piano.notes.append(note)
        #     curr_notes[event.value] = -1
        if event.type == "TIME_SHIFT":
            curr_time += event.value / 1000
        # if event.type == "SET_VELOCITY":
        #     curr_vel = event.value
    midi_data.instruments.append(piano)
    midi_data.write(fname)