import argparse
import bittensor
import torch
import torch.nn.functional as F

from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer,AutoConfig
from torch.nn.utils.rnn import pad_sequence
from bittensor.utils.tokenizer_utils import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets

from loguru import logger; logger = logger.opt(colors=True)

class server(torch.nn.Module):
    def __init__(self, 
                config: 'bittensor.config' = None,
                pretrained: bool = None,
                model_name: str = None,
                padding: bool =None, 
                interpolate: bool =None,
                inter_degree: str = None,
                model = None,
                tokenizer = None,
                mapping_function = None,
                token_remap = None,
                checking= None):
        r"""" Creates a server that serves up a pretrained miner on the bittensor network
        Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.server.config()
                pretrained (:obj:bool , `optional`):
                    if the model should pretrained or not
                model_name (:obj:string , `optional`):
                    name of the pretrained model from huggingface to use
                padding (:obj:bool, `optional`):
                    If the server should pad out to match the hidden units that the bittensor network is using
                    If set to False, it will instead create a mapping layer to do the same thing.
                interpolate (:obj:bool, `optional`):
                    If the server should interpolate between sequence length differences.
                    If set to false, there should be a mapping function that takes care of the differnces
                inter_degree (:obj:str, `optional`):
                    The Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)
                model (:obj:torch.module, `optional`):
                    Overrides the huggingface pretrained model with your own pretrained model
                tokenizer (:obj:huggingface.tokenizer, `optional`):
                    Overrides the huggingface tokenizer with your tokenizer
                mapping_function (:obj:Callable, `optional`):
                    Custom mapping function that maps between sequence length differences between tokenizers
                token_remap (:obj:Callable, `optional`):
                    Custom function that maps between tokenizers (defaults to self.remapping_token)
        """
        super(server, self).__init__()
        if config == None: config = server.config()
        self.config = config;print(config)
        
        #setting up pretrained model
        self.model_name = model_name if model_name != None else config.neuron.model_name
        self.pretrained = pretrained if pretrained != None else config.neuron.pretrained
        if self.pretrained == True:
            self.pre_model = model if model != None else AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = tokenizer if tokenizer != None else AutoTokenizer.from_pretrained(self.model_name)
        elif self.pretrained == False:
            model_config = AutoConfig.from_pretrained(self.model_name)
            model_config.vocab_size= bittensor.__vocab_size__
            self.pre_model = model if model != None else AutoModel.from_config(model_config)
            self.tokenizer = bittensor.tokenizer()

        if self.config.neuron.training:
            self.pre_model.train()
        elif self.config.neuron.autocast and self.device == 'cuda':
            self.pre_model.half()
        else:
            self.pre_model.eval()

        self.to_translation_map = get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = get_translation_map(self.std_tokenizer, self.tokenizer)
        self.split_map_cache = {}

        #parameters of the models
        self.final_dim =  bittensor.__network_dim__
        self.pre_dimension = self.pre_model.config.hidden_size
        self.device = config.neuron.device
        self.padding = padding if padding != None else config.neuron.padding
        self.interpolate = interpolate if interpolate != None else config.neuron.interpolate
        self.inter_degree = inter_degree if inter_degree != None else config.neuron.inter_degree
        self.checking = checking if checking != None else config.neuron.checking
        self.mapping_function= mapping_function
        self.token_remap = token_remap if token_remap != None else self.remapping_token

        if self.config.neuron.padding == False:
            self.mapping = torch.nn.Linear( self.pre_dimension, self.final_dim)

        self.decoder = torch.nn.Linear( self.final_dim, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
        self.outputs_cache = None
        self.gradients_cache = None

        #checking if the parameters of the server makes sense
        if self.checking and pretrained == True:
            self.check()
        
        # -- keeps track of gradients applied
        self.backward_gradients = 0 
        
    def forward(self, inputs,tokenizer=None):
        """
            Forward pass through the whole server model. Returns the loss and decoded predictions.

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer (:obj:'huggingface.tokenizer', optional):
                    The tokenizer which was used to tokenize the inputs
             Returns:
                loss (:obj:`torch.FloatTensor`):
                    MLM loss from the inputs
                decoded_targets (:obj:`torch.FloatTensor`):
                    Decoded predictions of the next token in the sentence.

        """
        decoded_targets = self.decoder(self.encode_forward(inputs,tokenizer))
        
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 

        return loss, decoded_targets
    
    def encode_forward(self,inputs,tokenizer=None):
        r""" Forward pass through the pretrained model and possible mappings between hidden units. 
             The response tensor should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        sen_len = inputs.size()
        inputs = self.token_remap(inputs,tokenizer).to(self.device)
        if self.config.neuron.training:
            pre_hidden = self.pre_model(inputs).last_hidden_state
        elif self.config.neuron.autocast and self.device == 'cuda':
            pre_hidden = self.pre_model(inputs).last_hidden_state
        else:
            with torch.no_grad():
                pre_hidden = self.pre_model(inputs).last_hidden_state
        
        if self.interpolate and sen_len[1] != pre_hidden.size()[1]:
            down= F.interpolate(pre_hidden.unsqueeze(1),size=[sen_len[1],pre_hidden.size()[2]],mode=self.inter_degree).squeeze(1)
        elif self.mapping_function:
            down = self.mapping_function(pre_hidden)
        else:
            down = pre_hidden


        if self.padding:
            padding_l = (self.final_dim-self.pre_dimension)//2
            padding_r = (self.final_dim-self.pre_dimension) - padding_l
            encoded_hidden = F.pad(down, (padding_l, padding_r),  "constant", 0)
        else:
            encoded_hidden = self.mapping(down)
        return encoded_hidden

    def remapping_token(self,input, old_tokenizer=None):
        r""" Default remapping of tokenizers; decodes the message and then remaps the message using a new tokenizer
            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                old_tokenizer ( huggingface.tokenizer, `required`):
                    The tokenizer which was used to tokenize the input  (defaults to bittensor tokenizer if none is given)
        """
        if old_tokenizer == None:
            old_tokenizer = bittensor.tokenizer()
        new_data = []
        for i in range(input.shape[0]):
            decoded = old_tokenizer.decode(input[i]) 
            hugging = self.tokenizer(decoded)
            new_data += [torch.LongTensor(hugging.input_ids)]
        new_data = pad_sequence(new_data,batch_first=True)
        return new_data

    def encode_forward_causallm(self, token_batch, tokenizer=None, encode_len=bittensor.__network_dim__):
        r""" Forward pass through the pretrained model and possible mappings between hidden units.
             The response tensor should be the hidden units computed using the local context and
             with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    torch inputs to be forward processed, [batch_size, sequence_len]
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs
                encode_len ( :obj:`int`, `optional`):
                    logit encoding length, default bittensor.__network_dim__ length

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        tokens = self.remapping_token_causallm(token_batch, tokenizer)  # remap to server tokenizer

        if self.config.neuron.training or (self.config.neuron.autocast and self.device[:4] == 'cuda'):
            pre_model_output = self.pre_model(input_ids=tokens['input_ids'],
                                              attention_mask=tokens['attention_mask'],
                                              output_hidden_states=True)
        else:
            with torch.no_grad():
                pre_model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                  attention_mask=tokens['attention_mask'],
                                                  output_hidden_states=True)

        pre_logits = pre_model_output.logits

        with torch.no_grad():
            probs_std = translate_logits_to_probs_std(pre_logits,
                                                      tokens['offset_mapping'], tokens['offset_mapping_std'],
                                                      self.tokenizer, self.std_tokenizer,
                                                      self.split_map_cache,
                                                      self.to_translation_map, self.from_translation_map,
                                                      tokens['input_ids'], token_batch)
        probs_std = probs_std.to(self.device)
        logits_std = torch.log(probs_std + 1e-64)

        # with torch.no_grad():
        #     original_loss = self.get_loss_fct(pre_logits.cpu(), tokens['input_ids'])
        #     translated_loss = self.get_loss_fct(logits_std, token_batch.cpu())

        return logits_std

    def remapping_token_causallm(self, token_batch, std_tokenizer=None):
        r""" Tokenizer remapping; decodes the message and then remaps the message using a new tokenizer
            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    token_batch to be retokenized, [batch_size, sequence_len]
                std_tokenizer ( transformers.Tokenizer, `optional`):
                    The standard tokenizer which was used to tokenize the input.
        """
        if std_tokenizer is None:
            std_tokenizer = self.std_tokenizer

        text_batch = std_tokenizer.batch_decode(token_batch)  # decode tokens to original text
        result = translate_special_token_text(text_batch, std_tokenizer, self.tokenizer)  # translate special tokens
        to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch = result

        std_tokens = std_tokenizer(text_batch, return_offsets_mapping=True)  # encode again to get offsets mapping
        tokens = self.tokenizer(to_text_batch, return_offsets_mapping=True, add_special_tokens=False)

        # pad offsets so that special token offset widths match for continued correct alignment
        std_tokens['offset_mapping'] = pad_offsets(std_tokens['offset_mapping'], from_offsets_batch, pad_offsets_batch)
        tokens['offset_mapping'] = pad_offsets(tokens['offset_mapping'], to_offsets_batch, pad_offsets_batch)

        tokens['offset_mapping_std'] = std_tokens['offset_mapping']  # include std token info

        for key in ['input_ids', 'attention_mask']:  # form a torch batch tensor
            tokens[key] = pad_sequence([torch.LongTensor(tensor) for tensor in tokens[key]], batch_first=True)
            tokens[key] = torch.LongTensor(tokens[key])

        return tokens

    def get_loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Calculate loss_fct, CausalLM loss, next-token prediction loss.
            Args:
                logits (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, sequence_len, bittensor.__network_dim__]
                labels (:obj:`torch.LongTensor`, `required`):
                    [batch_size, sequence_len]

            Returns:
                loss (:obj:`torch.FloatTensor`):
                    scalar
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def check(self):
        r"""Checks the server settings
        """
        assert self.tokenizer.name_or_path == self.pre_model.name_or_path, 'incorrect model ({}) and tokenizer ({})'.format(self.pre_model.name_or_path,self.tokenizer.name_or_path)
        if self.interpolate == False:
            assert self.mapping_function != None, 'Incorrect Settings; needs atleast one mapping function for sequence length changes'

    def save(self, path):
        try:
            state_dict = {
                'model': self.pretrained,
                'pretrained_model': self.pre_model.state_dict(), 
                'decoder': self.decoder.state_dict()
            }
            if self.padding == False:
                state_dict['mapping'] = self.mapping.state_dict()
            torch.save( state_dict, "{}/model.torch".format( path) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def load(self, path):
        try:
            state_dict=  torch.load("{}/model.torch".format( path ))
            if self.pretrained == state_dict['model']:
                self.pre_model.load_state_dict(state_dict['pretrained_model'], strict=False)
                self.decoder.load_state_dict(state_dict['decoder'])
                if self.padding == False:
                    self.mapping.load_state_dict(state_dict['mapping'])

                bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( path ))


        except Exception as e:
            logger.warning('No saved model found with error: {}', e)

    @staticmethod
    def config ():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.01)
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.model_name', type=str, help='pretrained model from hugging face',default='gpt2')
        parser.add_argument('--neuron.pretrained', action='store_false', help='if the model should be pretrained',default=True)
        parser.add_argument('--neuron.padding', action='store_false', help='To pad out final dimensions',default=True)
        parser.add_argument('--neuron.interpolate', action='store_false', help='To interpolate between sentence length',default=True)
        parser.add_argument('--neuron.inter_degree', type=str, help='Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)', default='nearest')
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='advanced_server')
        parser.add_argument('--neuron.checking', action='store_false', help='To check if server settings are correct',default=True)
        parser.add_argument('--neuron.restart', action='store_true', help='If True, train the neuron from the beginning', default=False)
        parser.add_argument('--neuron.blacklist.stake', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=10)
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch', default=10)
        parser.add_argument('--neuron.blacklist.time', type=int, help='how often a peer can query you (seconds) ', default=1)
        parser.add_argument('--neuron.training',  action='store_true', help='if the model should be training (increases memory load)', default=False)
        parser.add_argument('--neuron.autocast',  action='store_true', help='(experimental) autocasts the model to float16. Must require cuda', default=False)
        parser.add_argument('--neuron.blocks_per_set_weights', type=float, help='how often to set weights', default=100)
        parser.add_argument('--neuron.metagraph_sync', type=float, help='how often to sync the metagraph', default=100000)
        parser.add_argument('--neuron.blacklist_allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=False)


        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.prioritythreadpool.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.metagraph.add_args( parser )
        return bittensor.config( parser )
    
