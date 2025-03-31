import random
import pickle
from typing import NamedTuple
import torch
from torch.utils.data import Dataset
import dataclasses as dc
import copy
import time
MASK_IDX = 390
PAD_IDX = 391


@dc.dataclass
class MidiToken():

  '''
    Used to store a MIDI event.
    Valid (type, value) pairs are as follows:
    NOTE_ON: [0,127]
    NOTE_OFF: [0,127]
    TIME_SHIFT: [10,1000] (goes by 10s) ?
    SET_VELOCITY: [0,124] (goes by 4s)
    START: [0]
    STOP: [0]
    MASK: [0] ?
    PADDING: [0] ?
  '''
    
  type: str
  value: int
  time: float


  def __post_init__(self):
    self.value = int(self.value)
    self.time = float(self.time)
    self.type = str(self.type)


  def key_mapping(event):
    '''
    Given a MIDI event, return a unique index.

    '''
    # print('event_value', event.value)
    # print('even_type', event.type)

    if event.type == "NOTE_ON":
      return list(range(0,128))[event.value]
    if event.type == "NOTE_OFF":
      return list(range(128,256))[event.value]
    if event.type == "TIME_SHIFT":
      return list(range(256,356))[int(event.value // 10) - 1]
    if event.type == "SET_VELOCITY":
      if event.value>127:
        return list(range(356,388))[int(127 // 4)]
      return list(range(356,388))[int(event.value // 4)]
    if event.type == "START":
      return 388
    if event.type == "STOP":
      return 389
    if event.type == "MASK":
      return MASK_IDX
    if event.type == "PADDING":
      return PAD_IDX
    if event.type == "BEAT":
      return 392
    if event.type == "REM_TIME":
      return list(range(393,893))[int(event.value // 10)]
    if event.type == "BLANK":
      return 894
    if event.type == "PEDAL_ON":
      return 895
    if event.type == "PEDAL_OFF":
      return 896


  def tok_mapping(token):
    '''
    Given a MIDI token index, return the associated MIDI event.
    '''
    if torch.is_tensor(token):
        token = token.item()
    if token >=0 and token < 128:
        return MidiToken("NOTE_ON", token, 0)
    if token >= 128 and token < 256:
        return MidiToken("NOTE_OFF", token-128,0)
    if token >= 256 and token < 356:
        return MidiToken("TIME_SHIFT", ((token-256) * 10) + 10,0)
    if token >= 356 and token < 388:
        return MidiToken("SET_VELOCITY", (token-356) * 4,0)
    if token == 388:
        return MidiToken("START", 0,0)
    if token == 389:
        return MidiToken("STOP", 0,0)
    if token == MASK_IDX:
        return MidiToken("MASK", 0,0)
    if token == PAD_IDX:
        return MidiToken("PADDING", 0,0)  
    if token == 392:
      return MidiToken("BEAT", 110, 0)
    if token >= 393 and token < 893:
      return MidiToken("REM_TIME", ((token-393) * 10) ,0)
    if token == 894:
      return MidiToken("BLANK", 0,0)  
    if token == 895:
      return MidiToken("PEDAL_ON", 0,0)  
    if token == 896:
      return MidiToken("PEDAL_OFF", 0,0)  


def valid_timeshift(event):
  output = []
  ## event.value should be multiple of 10
  residual = event.value % 10
  if residual:
    new_value = event.value + (10 - residual)
  else:
    new_value = event.value



  #split to whole 1000 timeshifts
  num_time = new_value // 1000
  st =  event.time - (new_value / 1000) 
  for i in range(num_time):
    st += 1
    output.append(MidiToken("TIME_SHIFT", 1000,round(st,2)))
    
  st += (new_value % 1000) /1000
  output.append(MidiToken("TIME_SHIFT",new_value % 1000 , round(st,2)))


  # check if value is 0
  for i, ev in enumerate(output):
    if ev.value == 0:
      output.pop(i)

  return output



def key_mapping(event):
    '''
    Given a MIDI event, return a unique index.

    '''
    # print('event_value', event.value)
    # print('even_type', event.type)

    if event.type == "NOTE_ON":
      return list(range(0,128))[event.value]
    if event.type == "NOTE_OFF":
      return list(range(128,256))[event.value]
    if event.type == "TIME_SHIFT":
      return list(range(256,356))[int(event.value // 10) - 1]
    if event.type == "SET_VELOCITY":
        if event.value> 127:
            return list(range(356,388))[int(127 // 4)]
        return list(range(356,388))[int(event.value // 4)]
    if event.type == "START":
      return 388
    if event.type == "STOP":
      return 389
    if event.type == "MASK":
      return MASK_IDX
    if event.type == "PADDING":
      return PAD_IDX
    if event.type == "BEAT":
      return 392
    if event.type == "REM_TIME":
      return list(range(393,893))[int(event.value // 10)]
    if event.type == "BLANK":
      return 894
    if event.type == "PEDAL_ON":
      return 895
    if event.type == "PEDAL_OFF":
      return 896



def tok_mapping(token):
    '''
    Given a MIDI token index, return the associated MIDI event.
    '''
    if torch.is_tensor(token):
        token = token.item()
    if token >=0 and token < 128:
        return MidiToken("NOTE_ON", token,0)
    if token >= 128 and token < 256:
        return MidiToken("NOTE_OFF", token-128,0)
    if token >= 256 and token < 356:
        return MidiToken("TIME_SHIFT", ((token-256) * 10) + 10,0)
    if token >= 356 and token < 388:
        return MidiToken("SET_VELOCITY", (token-356) * 4 ,0)
    if token == 388:
        return MidiToken("START", 0 ,0)
    if token == 389:
        return MidiToken("STOP", 0 ,0)
    if token == MASK_IDX:
        return MidiToken("MASK", 0 ,0) 
    if token == PAD_IDX:
        return MidiToken("PADDING", 0 ,0)  
    if token == 392:
      return MidiToken("BEAT", 110, 0)
    if token >= 393 and token < 893:
      return MidiToken("REM_TIME", ((token-393) * 10) ,0)
    if token == 894:
      return MidiToken("BLANK", 0,0)  
    if token == 895:
      return MidiToken("PEDAL_ON", 0,0)  
    if token == 896:
      return MidiToken("PEDAL_OFF", 0,0)  
class MidiDataset(Dataset):
    '''
    MIDI sequences used in Music Transformer without preprocessing.
    Data augmentation includes pitch transpose and time stretch.
    The tokens are encoded as follows:
      0-127 = NOTE_ON
    128-255 = NOTE_OFF
    256-355 = TIME_SHIFT
    356-387 = SET_VELOCITY
    388-389 = START/STOP
    '''
    def __init__(self, filename, train = True, seq_len=512, train_split=0.8): 
        self.filename = filename
        self.seq_len = seq_len
        self.train_split = train_split 
        with open(filename, 'rb') as f:
            self.content = pickle.load(f)
        if train:
            self.content = self.content[:int(self.train_split*len(self.content))]
            
        else:
            self.content = self.content[int(self.train_split*len(self.content)):]

        # print(self.content)
        print("Num-songs", len(self.content))
        for i in range(len(self.content)):
            self.content[i]['input'] = [MidiToken("START", 0, 0)] + self.content[i]['input'] + [MidiToken("STOP", 0, self.content[i]['input'][-1].time)] # not sure how to change this line of code
            
        self.content = [s for s in self.content if len(s['input'])>=seq_len]
        print("Num-songs after discarding",len(self.content))

        self.num_songs = len(self.content)
        self._weights = torch.tensor([len(self.content[i]['input']) for i in range(self.num_songs)], dtype=torch.int).cuda()      

    def augment(self, seq, pitch_change, time_stretch, velocity_change):
        # data augmentation: pitch transposition
        # pitch_change = random.randint(-3,3)
        seq = [MidiToken(tok.type, tok.value + pitch_change, tok.time) if tok.type in ("NOTE_ON", "NOTE_OFF") and tok.value != 110 else tok for tok in seq]
        # data augmentation: time stretch
        # time_stretch = random.choice([0.95, 0.975, 1.0, 1.025, 1.05])
        seq = [MidiToken(tok.type, int(min(max((((time_stretch * tok.value) + 5) // 10) * 10, 10), 1000)), tok.time) if tok.type == "TIME_SHIFT" else tok for tok in seq]

        #data augmentation: velocity change
        # velocity_change = (random.randint(-3,3)) * 4
        seq = [MidiToken(tok.type, int(min(max(tok.value + velocity_change, 0), 127)), tok.time) if tok.type == "SET_VELOCITY" else tok for tok in seq] 

        return seq



    def get_batch(self, batch_size):
        i = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp = torch.full((batch_size, self.seq_len), PAD_IDX, device=device).long()
        trg = torch.full((batch_size, self.seq_len), PAD_IDX, device=device).long()
        # task_token = torch.tensor([[0],[0],[0],[0],
        #                            [1],[1],[1],[1], 
        #                            [2],[2],[2],[2], 
        #                            [3],[3],[3],[3],
        #                            [4],[4],[4],[4]], device=device)  
        task_token = torch.tensor([[3],[3],[3],[3],
                                   [3],[3],[3],[3],
                                   [3],[3],[3],[3],
                                   [3],[3],[3],[3],
                                   [3],[3],[3],[3]], device=device)  
        task_token = task_token.type(torch.int64)
        name = []
        tempo = []


        while i < batch_size:
          song_idx = torch.multinomial(self._weights.float(), 1, replacement=True)
          seq_lens = torch.randint(self.seq_len//2, self.seq_len+1, size=(1,)) - 2
          song_pos = [torch.randint(0, self._weights[song_idx[j]].item()-seq_lens[j]+1, size=(1,))[0] for j in range(1)]          
        
          
          if self.content[song_idx[0]]['name'] != "Glinka_The_Lark_Denisova10M.mid":

            x = self.content[song_idx[0]]['input'][song_pos[0]:song_pos[0]+seq_lens[0]].copy()
            name.append(self.content[song_idx[0]]['name'])
            tempo.append(self.content[song_idx[0]]['tempo'])

            pitch_change = random.randint(-3,3)
            time_stretch = random.choice([0.95, 0.975, 1.0, 1.025, 1.05])
            velocity_change = (random.randint(-3,3)) * 4

            if task_token[i] == 0: #music2beat:
              start = x[0].time
              end = x[-1].time
              if x[0].type == "TIME_SHIFT":
                start = x[0].time - (x[0].value / 1000)

              beat_times = torch.tensor([beat.time for beat in self.content[song_idx[0].item()]['beat']], device=device)
              beat_types = torch.tensor([1 if beat.type != "TIME_SHIFT" else 0 for beat in self.content[song_idx[0].item()]['beat']], device=device)


              # Find the start index
              start_condition = (beat_times >= start) & (beat_types == 1)
              b_start_indices = torch.where(start_condition)[0]

              if b_start_indices.numel() > 0:
                  b_index_start = b_start_indices[0].item() + 1
              else:
                  b_index_start = 0  # or some other default value

              # Find the end index
              end_condition = (beat_times <= end) & (beat_types == 1)
              b_end_indices = torch.where(end_condition)[0]

              if b_end_indices.numel() > 0:
                  b_index_end = b_end_indices[-1].item()
              else:
                  b_index_end = 0

              b = self.content[song_idx[0].item()]['beat'][b_index_start: b_index_end + 1].copy()
              if len(b) == 0 :
                print(self.content[song_idx[0]]['name'])
                continue
              else:
                if  0.01 < (b[0].time - start) :
                  temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((b[0].time - start) * 1000), time=b[0].time))
                  for indx in range(len(temp)):
                    b.insert(indx, temp[indx])


              if 0.01 < (end - b[-1].time) :
                temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((end - b[-1].time) * 1000), time=end)) 
                b += temp

 
              b.insert(0,MidiToken("START", 0, 0))   
              b.append(MidiToken("STOP", 0, 0))

              x = self.augment(x, pitch_change, time_stretch, velocity_change)
              b = self.augment(b, 0 , time_stretch, 0)
              # print(x)

              inp[i, :len(x)] = torch.tensor(list(map(key_mapping, x)))  # pitch tempo velocity
              trg[i,:len(b)] = torch.tensor(list(map(key_mapping, b))[:512])   # tempo
              i += 1


            elif task_token[i] == 1: #music2music:
              x = self.augment(x, pitch_change, time_stretch, velocity_change)
            
              inp[i, :len(x)] = torch.tensor(list(map(key_mapping, x))) # pitch tempo velocity
              x.insert(0,MidiToken("START", 0, 0))   
              x.append(MidiToken("STOP", 0, 0))

              # print(x)
              trg[i,:len(x)] = torch.tensor(list(map(key_mapping, x)))  # pitch tempo velocity
              i += 1


            elif task_token[i] == 2: #music2musicbeatremtime
              # aligning integrated sequence and music
              start = x[0].time
              end = x[-1].time
              if x[0].type == "TIME_SHIFT":
                start = x[0].time - (x[0].value / 1000)
                  
              c_index_start = 0
              c_index_end = 0
              intg_times = torch.tensor([intg.time for intg in self.content[song_idx[0].item()]['intg']], device=device)
              intg_types = torch.tensor([1 if intg.type != "TIME_SHIFT" else 0 for intg in self.content[song_idx[0].item()]['intg']], device=device)

              c_index_start = torch.where((intg_times >= start) & (intg_types == 1))[0][0].item()
              c_index_end = torch.where((intg_times <= end) & (intg_types == 1))[0][-1].item()

              c = self.content[song_idx[0].item()]['intg'][c_index_start: c_index_end + 1].copy()
              if len(c) == 0 :
                print(self.content[song_idx[0]]['name'])
                if  0.01 < (c[0].time - start) :
                  temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((c[0].time - start) * 1000), time=c[0].time))
                  for indx in range(len(temp)):
                    c.insert(indx, temp[indx])


              if 0.01 < (end - c[-1].time) :
                temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((end - c[-1].time) * 1000), time=end)) 
                c += temp
              # adding new REM_TIME tokens to the intg sequence to get an idea where is next Beat
              c_copy = []
              flag1 = True
              for f in range(len(c)-1):
                if c[f].type == "TIME_SHIFT":
                  c_copy.append(c[f])
                  if c[f+1].type == "BEAT":
                    c_copy.append(MidiToken("REM_TIME",0,0))
                  elif c[f+1].type != "BEAT":
                    time_value = 0
                    for n in range(f+1, len(c)):
                      if c[n].type == "TIME_SHIFT":
                        time_value += c[n].value
                      elif c[n].type == "BEAT":
                        c_copy.append(MidiToken("REM_TIME",time_value,0))
                        if time_value >= 5000:
                          flag1 = False
                        break
                else:
                  c_copy.append(c[f])
              c_copy.append(c[-1])  


              c_copy.insert(0,MidiToken("START", 0, 0))   
              c_copy.append(MidiToken("STOP", 0, 0))

              x = self.augment(x, pitch_change, time_stretch, velocity_change)
              c_copy = self.augment(c_copy, pitch_change, time_stretch, velocity_change)
              # print(x)
              # print(c_copy)
              if flag1:
                inp[i,:len(x)] = torch.tensor(list(map(key_mapping, x))) # pitch tempo velocity
                trg[i,:len(c_copy)] = torch.tensor(list(map(key_mapping, c_copy))[:512]) # pitch tempo velocity
                if trg[i,-1 ] != PAD_IDX:
                    trg[i,-1] = 389
                    
                i += 1 
              else:
                 continue




            elif task_token[i] == 3: #music2musicbeat
              # aligning integrated sequence and music
              start = x[0].time
              end = x[-1].time
              if x[0].type == "TIME_SHIFT":
                start = x[0].time - (x[0].value / 1000)
                  

              c_index_start = 0
              c_index_end = 0
              intg_times = torch.tensor([intg.time for intg in self.content[song_idx[0].item()]['intg']], device=device)
              intg_types = torch.tensor([1 if intg.type != "TIME_SHIFT" else 0 for intg in self.content[song_idx[0].item()]['intg']], device=device)

              c_index_start = torch.where((intg_times >= start) & (intg_types == 1))[0][0].item()
              c_index_end = torch.where((intg_times <= end) & (intg_types == 1))[0][-1].item()

              c = self.content[song_idx[0].item()]['intg'][c_index_start: c_index_end + 1].copy()
              if len(c) == 0 :
                print(self.content[song_idx[0]]['name'])
                if  0.01 < (c[0].time - start) :
                  temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((c[0].time - start) * 1000), time=c[0].time))
                  for indx in range(len(temp)):
                    c.insert(indx, temp[indx])


              if 0.01 < (end - c[-1].time) :
                temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((end - c[-1].time) * 1000), time=end)) 
                c += temp
              c.insert(0,MidiToken("START", 0, 0))   
              c.append(MidiToken("STOP", 0, 0))

              x = self.augment(x, pitch_change, time_stretch, velocity_change)
              c = self.augment(c, pitch_change, time_stretch, velocity_change)
              # print(c_copy)
              inp[i,:len(x)] = torch.tensor(list(map(key_mapping, x))[:512])
              trg[i,:len(c)] = torch.tensor(list(map(key_mapping, c))[:512])
              if trg[i,-1 ] != PAD_IDX:
                  trg[i,-1] = 389
              i += 1  



            elif task_token[i] == 4: #denoising 
              start = x[0].time
              end = x[-1].time
              if x[0].type == "TIME_SHIFT":
                start = x[0].time - (x[0].value / 1000)

              c_index_start = 0
              c_index_end = 0
              intg_times = torch.tensor([intg.time for intg in self.content[song_idx[0].item()]['intg']], device=device)
              intg_types = torch.tensor([1 if intg.type != "TIME_SHIFT" else 0 for intg in self.content[song_idx[0].item()]['intg']], device=device)

              c_index_start = torch.where((intg_times >= start) & (intg_types == 1))[0][0].item()
              c_index_end = torch.where((intg_times <= end) & (intg_types == 1))[0][-1].item()

              c = self.content[song_idx[0].item()]['intg'][c_index_start: c_index_end + 1].copy()
              if len(c) == 0 :
                print(self.content[song_idx[0]]['name'])
                if  0.01 < (c[0].time - start) :
                  temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((c[0].time - start) * 1000), time=c[0].time))
                  for indx in range(len(temp)):
                    c.insert(indx, temp[indx])


              if 0.01 < (end - c[-1].time) :
                temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((end - c[-1].time) * 1000), time=end)) 
                c += temp


              start = x[0].time
              end = x[-1].time
              if x[0].type == "TIME_SHIFT":
                start = x[0].time - (x[0].value / 1000)
                  
              n_index_start = 0
              n_index_end = 0
              nintg_times = torch.tensor([nintg.time for nintg in self.content[song_idx[0].item()]['nintg']], device=device)
              nintg_types = torch.tensor([1 if nintg.type != "TIME_SHIFT" else 0 for nintg in self.content[song_idx[0].item()]['nintg']], device=device)

              n_index_start = torch.where((nintg_times >= start) & (nintg_types == 1))[0][0].item()
              n_index_end = torch.where((nintg_times <= end) & (nintg_types == 1))[0][-1].item()

              n = self.content[song_idx[0].item()]['nintg'][n_index_start: n_index_end + 1].copy()
              if len(n) == 0 :
                print(self.content[song_idx[0]]['name'])
                if  0.01 < (n[0].time - start) :
                  temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((n[0].time - start) * 1000), time=n[0].time))
                  for indx in range(len(temp)):
                    n.insert(indx, temp[indx])


              if 0.01 < (end - n[-1].time) :
                temp = valid_timeshift(MidiToken(type='TIME_SHIFT', value = int((end - n[-1].time) * 1000), time=end)) 
                n += temp


              c.insert(0,MidiToken("START", 0, 0))   
              c.append(MidiToken("STOP", 0, 0))

              n = self.augment(n, pitch_change, time_stretch, velocity_change)
              c = self.augment(c, pitch_change, time_stretch, velocity_change)
              inp[i,:len(n)] = torch.tensor(list(map(key_mapping, n))[:512])
              trg[i,:len(c)] = torch.tensor(list(map(key_mapping, c))[:512])
              if trg[i,-1 ] != PAD_IDX:
                  trg[i,-1] = 389
              i += 1  

            
          else:
             continue
          



        trg = torch.cat((task_token, trg[:, :-1]), 1)
        # print(inp)
        # print(trg)



        return {
            "inp": inp,
            "trg": trg,
            "name" : name,
            "tempo": tempo,

        }

# if __name__ == "__main__":
#     print('hi', flush=True)
#     valid = MidiDataset(
#         filename='/h/hana/Documents/ASAP_Arxiv/dataset/ASAP_valid.pickle',
#         train=True,
#         seq_len=512,
#         train_split=1)
#     output_folder = '/h/hana/Documents/NextToken/Nocturne2023_beat/5task/test_validation'
#     for i in range(2):
#       s = time.time()
#       data = valid.get_batch(20)
#       print(data)
#       for j in range(20):
#         pred_seq = list(map(MidiToken.tok_mapping, data['inp'][j].cpu()))
#         dat2mid_anna(pred_seq, data['tempo'][j],
#                                       fname=output_folder + "track" + str(preds) +
#                                       "/output_" + str(n) + "_" + str(i + 1) +
#                                       ".mid")
#         print(time.time()-s)
