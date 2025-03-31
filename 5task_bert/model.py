import torch
import torch.nn as nn

from transformers import AutoModel, BertConfig, BertModel

MASK_IDX = 390
PAD_IDX = 391

class BertModel(nn.Module):
    def __init__(self, args):
        super(BertModel, self).__init__()
        self.pretrained = args.pretrained

        self.bert = AutoModel.from_pretrained(f"google/bert_uncased_L-{args.layers}_H-{args.embed_dim}_A-{args.embed_dim // 64}", )
        if not self.pretrained:
            self.bert.apply(self.bert._init_weights)


        self.token_embedding = nn.Embedding(5, args.embed_dim)
        self.embbeding = self.bert.embeddings
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_dim, nhead=args.embed_dim // 64,)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.layers)

        self.fc = nn.Linear(args.embed_dim, args.vocab_size)

    def forward(self, inp, target, inp_pad_mask = None,trg_pad_mask = None ):
        mask = self.generate_square_subsequent_mask(target.size(0), target.size(1)).to(inp.device)

        if inp_pad_mask is None:
             inp_pad_mask = (inp == PAD_IDX).to(inp).bool()

        if trg_pad_mask is None:
            trg_pad_mask = (target == PAD_IDX).to(target).bool()


        enc = self.bert(input_ids=inp).last_hidden_state

        task_token = target[:,0]
        task_token = self.token_embedding(task_token)
        target = self.embbeding(target[:,1:])
        target = torch.cat((task_token.unsqueeze(1), target), dim=1)


        # print("src shape",enc.shape , " target shape ", target.shape ," mask shape", mask.shape)
        out = self.decoder(target.transpose(0,1), memory = enc.transpose(0,1), tgt_mask = mask, 
                           tgt_key_padding_mask = trg_pad_mask, memory_key_padding_mask = inp_pad_mask
                           )
        out = out.transpose(0,1)
        out = self.fc(out)
        return out

    def generate_square_subsequent_mask(self, n, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # (L, L)
        # mask = mask.float().unsqueeze(0).repeat(n, 1, 1) # (B, L, L)
        return mask


if __name__ == "__main__":
    from argparse import Namespace
    args_dict = {
        "pretrained": 1,
        "layers": 6,
        "embed_dim": 512,
        "vocab_size": 897,
    }
    args = Namespace(**args_dict)
    model = BertModel(args)
    inp = torch.randint(0, 896, (1,512), dtype=torch.long)
    token = torch.randint(0, 4, (1,1), dtype=torch.long)
    print(token)
    target = torch.randint(0, 896, (1,511), dtype=torch.long)
    target = torch.cat((token, target),1)
    # print(trg)
    # print(inp.shape, mean.shape, var.shape)
    out = model(inp, target)
    print(out.shape)
