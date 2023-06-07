import torch
import torch.nn as nn
from decision_transducer.models.encoders import Encoder, get_lookahead_mask

class JoinNet(torch.nn.Module):

    def __init__(self, hidden_size, n_layers = 1, nhead = 2, pdrop = 0.1, pre_norm = False,  norm_joint = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_joint = norm_joint

        if self.norm_joint:
            self.norm1 = torch.nn.LayerNorm(hidden_size)

        self.join_enc = Encoder(hidden_size, n_layers = n_layers, nhead = nhead, pdrop = pdrop, pre_norm = False)

        self._init_params()
    
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
    
    def forward(self, states, actions, pad_mask = None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        stacked_inputs = torch.stack(
                (states, actions), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        if self.norm_joint:
            stacked_inputs = self.norm1(stacked_inputs)
        
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_pad_mask = torch.stack(
            (pad_mask, pad_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        causal_mask = get_lookahead_mask(stacked_pad_mask )

        x = self.join_enc(stacked_inputs, causal_mask, stacked_pad_mask)
        # the 0 dim is batch, 1 dim is state,action
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # retrieve state
        return x[:,0]
    
    def forward_all(self, rtgs, states, actions, pad_mask = None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        stacked_inputs = torch.stack(
                (rtgs, states, actions), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        if self.norm_joint:
            stacked_inputs = self.norm1(stacked_inputs)
        
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_pad_mask = torch.stack(
            (pad_mask, pad_mask, pad_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        causal_mask = get_lookahead_mask(stacked_pad_mask )

        x = self.join_enc(stacked_inputs, causal_mask, stacked_pad_mask)
        # the 0 dim is batch, 1 dim is state,action
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # retrieve state
        return x[:,1]

    # concatenation does not work
    # def forward(self, states, actions, pad_mask):
    #     states = self.norm1(states)
    #     actions = self.norm2(actions)
    #     fuse = torch.cat([states,actions], dim = -1)
    #     states =  states + self.dropout1( self.w2( self.gelu( self.w3(fuse) ) ) )
    #     return self.norm3(states)

    # def forward(self, states, actions, pad_mask):
    #     attn_mask = get_lookahead_mask(actions)
    #     states_0 = states
    #     states_1, _ = self.attn(
    #         query = actions,
    #         key = states,
    #         value = states,
    #         key_padding_mask = pad_mask,
    #         attn_mask = attn_mask
    #         )
    #     final = self.w3( self.tanh( self.w1(states_0) + self.w2(states_1)) )
    #     return self.norm1(final)

    # def forward(self, states, actions, pad_mask):
    #     attn_mask = get_lookahead_mask(actions)
    #     states_0 = states
    #     states_1, _ = self.attn(
    #         query = states_0,
    #         key = actions,
    #         value = actions,
    #         key_padding_mask = pad_mask,
    #         attn_mask = attn_mask
    #         )
    #     # add & norm
    #     states = states + self.dropout1(states_1)
    #     states = self.norm1(states)

    #     # FFN
    #     states_0 = states
    #     states_1 = self.w2( self.gelu( self.w1(states_0)))

    #     # add & norm
    #     states = states + self.dropout2( states_1 )
    #     return self.norm2(states)



    # def forward(self, a,b, pad_mask):
    #     return a
        # attn_mask = get_lookahead_mask(b)
        # return self.gelu( self.w1(a) + self.w2(b) )
        # return self.tanh( self.w1(a) + self.w2(b) )
        # c = torch.cat([a,b], dim = -1)
        # return self.tanh( self.w3(c) )
        # c,_ = self.attn(
        #     query = a,
        #     key = b,
        #     value = b,
        #     key_padding_mask = pad_mask,
        #     attn_mask = attn_mask
        #     )
        # 
        # return self.tanh( self.w2(c) )