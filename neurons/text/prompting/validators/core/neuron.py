# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import json
import math
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel
from transformers import AutoTokenizer

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific. Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
'''

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        full_path = os.path.expanduser('{}/netuid{}/{}'.format( config.logging.logging_dir, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser( full_path )
        config.neuron.reward_path = os.path.expanduser( config.neuron.reward_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path, exist_ok = True)
        if not os.path.exists( config.neuron.reward_path + '/hf_ckpt.pt' ):
            os.makedirs( config.neuron.reward_path, exist_ok = True )
            os.system(
                f"wget -O { config.neuron.reward_path + '/hf_ckpt.pt'} \
                https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
            )
        if not config.neuron.dont_save_events:
            # Add custom event logger for the events.
            logger.level("EVENTS", no=38, icon="📝")
            logger.add( 
                config.neuron.full_path + "/" + "completions.log", 
                rotation=config.neuron.events_retention_size, serialize=True, enqueue=True, backtrace=False, diagnose=False, level="EVENTS", 
                format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message} | {extra[prompt]} {extra[completion]} {extra[uids]} {extra[all_uids]} {extra[rewards]}{extra[all_completions]} {extra[block]}"
            )


    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.base_prompt', type=str, help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument( '--neuron.question_prompt', type=str, help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument( '--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument( '--neuron.length_timeout_multiplier', type = int, help = 'Base timeout for all requests.', default = 0.01 )
        parser.add_argument( '--neuron.inference_topk', type = int, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument( '--neuron.training_topk', type = int, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument( '--neuron.training_timeout', type = int, help = 'Query timeout during training', default = 4 )
        parser.add_argument( '--neuron.inference_timeout', type = int, help = 'Query timeout during inference', default = 10 )
        parser.add_argument( '--neuron.inference_only', action = 'store_true', help = 'If set, training off and only inference will be served via axon.', default = False )
        parser.add_argument( '--neuron.axon_off', action = 'store_true', help = 'If set, the axon will be turned off.', default = False )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100000 )
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        parser.add_argument( '--neuron.epoch_length_override', type = int, help = 'Override the default timeout', default = -1 )
        parser.add_argument( '--neuron.dont_save_events', action = 'store_true', help = 'If set, we dont save events to a log file.', default = False )
        parser.add_argument( '--neuron.events_retention_size',  type = str,  help = 'Events retention size.', default = "2 GB" )
        parser.add_argument( '--neuron.no_reward_model', action = 'store_true', help = 'If set, we dont load the reward model instead use just the scores.', default = False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.logging.add_args( parser )
        GatingModel.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self ):
        self.config = neuron.config()
        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        
        self.device = torch.device( self.config.neuron.device )

        self.tokenizer = AutoTokenizer.from_pretrained( 'EleutherAI/gpt-j-6b' )

        self.alpha = 0.99
        # Reward model
        if not self.config.neuron.no_reward_model:
            bittensor.logging.info('Loading reward model')
            self.reward_model = RewardModel( model_path = 'EleutherAI/gpt-j-6b', device = self.config.neuron.device )
            for fpath in os.listdir( self.config.neuron.reward_path ):
                if fpath.endswith(".pt") or fpath.endswith(".bin"):
                    checkpoint = os.path.join( self.config.neuron.reward_path, fpath )
                    break
            ckpt_state = torch.load( checkpoint )
            self.reward_model.load_state_dict( ckpt_state )
            self.reward_model.eval()
            if self.device == "cuda":
                self.reward_model.half()
            self.reward_model.requires_grad_( False )
            self.reward_model.to( self.device )
            bittensor.logging.info('done loading reward model')



    def forward(
            self, 
            roles: List[ str ],
            messages: List[ str ],
            successful_completions: List[str],
        ) -> SimpleNamespace:

        flattened_message_for_reward = ''
        for role_i, message_i in list(zip(roles, messages)):
            if role_i != 'system': flattened_message_for_reward += message_i.strip() + '\n\n'
        full_completions_for_reward = [ 'Question: ' + flattened_message_for_reward + 'Answer: ' + comp.strip() for comp in successful_completions ]
        completions_for_reward = [comp.strip() for comp in successful_completions] 
        rewards = self.reward_model.reward( full_completions_for_reward, completions_for_reward, difference = True, shift = 3).detach().to( self.device )

        bittensor.logging.trace( 'rewards', rewards )
        print("Rewards", rewards)
        return rewards




from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
    roles = request.args.getlist("roles")
    messages = request.args.getlist("messages")
    successful_completions = request.args.getlist("successful_completions")

    # Do something with the arguments, if needed
    rewards = active_neuron.forward(roles=roles, messages=messages, successful_completions=successful_completions)
    
    return str(rewards)    

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    roles = ["user:"]
    messages= ["Who narrated the christmas tv special rudolph the red-nosed reindeer?"]
    successful_completions= ['The Christmas TV special "Rudolph the Red-Nosed Reindeer" was narrated by actor and comedian, Bill Cosby.']
    active_neuron = neuron()
    active_neuron.forward(roles=roles, messages=messages, successful_completions=successful_completions)

    app.run(host="0.0.0.0", port=8008)