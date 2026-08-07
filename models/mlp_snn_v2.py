"""
mlp_snn_v2.py

A four layer MLP spiking neural network with optional hidden-layer
spike compression and reconstruction.

Methods:
forward: Performs timestep-based spike propagation and optional compression.
"""

import torch
import torch.nn as nn
import snntorch as snn
from compression import compression
import config

class MLP_SNN_V2(nn.Module):
    def __init__(self, numInputs, numHidden1, numHidden2, numOutputs, numTimesteps, beta):
        super().__init__()

        self.num_inputs = numInputs
        self.num_hidden1 = numHidden1
        self.num_hidden2 = numHidden2
        self.num_outputs = numOutputs
        self.num_timesteps = numTimesteps
        self.beta = beta
    
        # Network Layers
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden1)      
        self.lif1 = snn.Leaky(beta=self.beta)

        self.fc2 = nn.Linear(self.num_hidden1, self.num_hidden2)
        self.lif2 = snn.Leaky(beta=self.beta)

        self.fc3 = nn.Linear(self.num_hidden2, self.num_outputs)
        self.lif3 = snn.Leaky(beta=self.beta)

    def forward(self, x, compressionMode = False, patternMiningFunc = None, numFsp = 1, pmt = 0.65):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record spikes
        spk1_rec = [] 
        spk3_rec = []
        mem3_rec = []

        # First half of the network
        cur1 = self.fc1(x)
        for step in range(self.num_timesteps):
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)

        hiddenSpikes = torch.stack(spk1_rec, dim=0)

        batchMetrics = None

        if compressionMode:
            compressedAer, shape, PATTERNLIST, SYMBOLIST, metrics = compression.compress(hiddenSpikes, patternMiningFunc, numFsp=numFsp, pmt=pmt, num_hidden=self.num_hidden1, minSup=config.CommonConfig.MINSUP_2)
            hiddenSpikes = compression.decompress(self, compressedAer, PATTERNLIST, SYMBOLIST, shape)
            batchMetrics = metrics
        
        # Second half of the network
        for step in range(self.num_timesteps):
            # Process through layer 2
            cur2 = self.fc2(hiddenSpikes[step])
            spk2, mem2 = self.lif2(cur2, mem2)

            # The input is the spike output from layer X (spkX)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif2(cur3, mem2)

            # Record the final layer's output
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0), batchMetrics