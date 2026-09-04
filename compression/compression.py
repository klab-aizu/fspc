"""
compression.py

Utilities for spike data conversion and compression.

Functions:
    convertToAER: Convert spike tensors to AER format.

    convertToSpike: Reconstruct spike tensors from AER data.

    mlextendFpmax: Mine maximal frequent patterns with FPMax.

    pamiFpgrowth: Mine frequent patterns with PAMI FP-Growth.

    pamiMaxFpgrowth: Mine maximal patterns with PAMI MaxFPGrowth.

    compress: Compress AER data using frequent patterns.
    
    decompress: Restore compressed AER data.
"""

import pandas as pd
import psutil
import os
import time
import math
import torch
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax
from PAMI.frequentPattern.basic import FPGrowth as PamiFPGrowth
from PAMI.frequentPattern.maximal import MaxFPGrowth as PamiMaxFPGrowth
from collections import defaultdict

# Convert spike data into AER (Address Event Representation) format
def convertToAER(spikeData):
    """
    Convert spike tensor data into Address Event Representation (AER) format.

    Args:
        spikeData: Binary spike tensor containing neuron activations.
            Expected format: (time, sample, neuron).

    Returns:
        list: AER representation containing spike events as tuples.
            Each tuple contains (timestep, sample, neuron).
    """

    print("Converting hidden spikes to AER data...")

    spikeIndices = spikeData.nonzero(as_tuple=False).cpu()

    # Convert directly from tensor rows to tuples
    aerData = [tuple(x.tolist()) for x in spikeIndices]

    print("Converted hidden spikes to AER data...")

    return aerData
    
# Convert AER (Address Event Representation) into spike data format
def convertToSpike(net, aerData, shape):
    """
    Convert Address Event Representation (AER) data back into spike tensor format.

    Args:
        net: snnTorch model used to determine the target computation device.
        aerData: AER spike event list containing (timestep, sample, neuron) tuples.
        shape: Desired output spike tensor shape.

    Returns:
        torch.Tensor: Binary spike tensor reconstructed from AER data.
    """

    targetDevice = net.fc1.weight.device

    spikeData = torch.zeros(shape, dtype=torch.float32, device=targetDevice)

    if len(aerData) == 0:
        return spikeData

    indices = torch.tensor(aerData, dtype=torch.long, device=targetDevice)

    spikeData[indices[:,0], indices[:,1], indices[:,2]] = 1.0

    return spikeData

# ___mlextend fpmax algorithm___
def mlextendFpmax(aerData, numFsp, minSup):
    """
    Extract maximal frequent neuron activation patterns using the mlxtend FPMax algorithm.

    Args:
        aerData: AER spike event list containing (timestep, sample, neuron) tuples.
        numFsp: Number of representative activation patterns to return.
        minSup: Minimum support threshold required for frequent patterns.

    Returns:
        tuple:
            fspList: List of extracted neuron activation patterns.
            runtimeEx: Execution time of the FPMax algorithm in seconds.
            rssEx: Additional memory usage during FPMax execution in bytes.
    """

    # Initialize FSP (Frequent Spike Pattern)
    fspList = []

    # Group neuron spikes by timestep and sample into transactions
    transactionDict = defaultdict(list)
    for timestep, sample, neuron in aerData:
        transactionDict[(timestep, sample)].append(neuron)

    # Convert grouped transactions into a list of transactions
    transactions = list(transactionDict.values())
    print(f"Total transactions: {len(transactions)}")

    # Transform transactions into a sparse one-hot encoded DataFrame
    transactionEncoder = TransactionEncoder()
    transactionEncoderArray = transactionEncoder.fit_transform(transactions)
    
    # Explicit bool dtype prevents mlxtend internal re-casting
    dataFrame = pd.DataFrame(
        transactionEncoderArray, 
        columns=transactionEncoder.columns_, 
        dtype=bool
    )

    print("Running mlxtend fpmax()...")
    startTime = time.time()

    # Get process object to track memory usage
    process = psutil.Process(os.getpid())

    # Memory usage before calling fpmax
    mem_before = process.memory_info().rss

    # Run mlxtend fpmax algorithm to find frequent neuron activation patterns
    frequentPatternsFpmax = fpmax(dataFrame, min_support=minSup, use_colnames=True)

    # Memory usage after calling fpmax
    mem_after = process.memory_info().rss

    # Calculate memory used during fpmax execution
    rssEx = mem_after - mem_before
    print("Analyzing patterns...")

    # Check if any frequent patterns were found
    if not frequentPatternsFpmax.empty:
        # Calculate the number of neurons in each pattern
        frequentPatternsFpmax["length"] = frequentPatternsFpmax["itemsets"].str.len()
        
        # Select the highest support pattern for each pattern size
        top_patterns_df = (
            frequentPatternsFpmax
            .sort_values(["length", "support"], ascending=[False, False])
            .groupby("length")
            .head(1)
        )

        # Select the final number of frequent spike pattern
        final_selection = top_patterns_df.head(numFsp)

        # Print the size of each selected pattern
        for fsp in final_selection["itemsets"]:
            fspList.append(set(map(int, fsp)))
        
        print(f"Itemset: {fspList}")

        # Print support values for selected patterns
        print(f"Moderate itemset size: {[len(x) for x in fspList]}")

        # Print support values for selected patterns
        print("Support (minsup):", [f"{s:.3f}" for s in final_selection["support"]])
    else:
        print("No patterns found.")

    # Calculate total runtime of fpmax analysis
    runtimeEx = time.time() - startTime
    print("Analysis complete.")

    # Display execution time and memory usage
    print(f"Elapsed time for mlextend fpmax algorithm: {runtimeEx:.3f} seconds")
    print(f"Memory used by fpmax call: {rssEx} bytes")

    # Return frequent spike patterns, runtime, and memory usage
    return fspList, runtimeEx, rssEx

    
# ___pami fpgrowth algorithm___
def pamiFpgrowth(aerData, numFsp, minSup):
    """
    Extract frequent neuron activation patterns using the PAMI FP-Growth algorithm.

    Args:
        aerData: AER spike event list containing (timestep, sample, neuron) tuples.
        numFsp: Number of representative activation patterns to return.
        minSup: Minimum support threshold required for frequent patterns.

    Returns:
        tuple:
            fspList: List of extracted frequent neuron activation patterns.
            runtimePami: Runtime reported by the PAMI FP-Growth algorithm in seconds.
            rssPami: Memory usage reported by the PAMI algorithm in bytes.
    """

    # Initialize FSP (Frequent Spike Pattern)
    fspList = []
    
    # Group neuron spikes by timestep, sample into transactions
    transactionDict = defaultdict(list)
    for timestep, sample, neuron in aerData:
        transactionDict[(timestep, sample)].append(neuron)
    transactions = list(transactionDict.values())
    
    print(f"Total transactions: {len(transactions)}")

    # Temp file storing transaction data
    tempFile = "transactionsTemp.txt"

    # Save transactions to file (PAMI input format)
    with open(tempFile, "w") as f:
        for txn in transactions:
            f.write(" ".join(map(str, txn)) + "\n")

    print("Running PAMI FPGrowth()...")
    obj = PamiFPGrowth.FPGrowth(tempFile, minSup = minSup, sep = " ")
    obj.mine()

    # Get results as DataFrame
    patternsDf = obj.getPatternsAsDataFrame()  # columns: ['Patterns', 'Support']

    # Add length column
    patternsDf["length"] = patternsDf["Patterns"].apply(lambda x: len(x.split()))

    # 1. Get a list of unique lengths present in the data
    unique_lengths = sorted(patternsDf["length"].unique())

    # 2. For each unique length, find the pattern with the highest support
    top_patterns_lengthwise = []

    for length in unique_lengths:
        # Filter for patterns of this specific length
        patterns = patternsDf[patternsDf['length'] == length]
        # Get the one with the highest support
        topPattern = patterns.iloc[0]
        top_patterns_lengthwise.append(topPattern)

    # 3. Create a new DataFrame from these top patterns
    top_patterns_df = pd.DataFrame(top_patterns_lengthwise)

    # 4. Sort these top patterns by their support value to get the best of the best
    top_patterns_df = top_patterns_df.sort_values(by=['length', 'Support'], ascending=[False, False])

    # 5. Select the final `numFsp` patterns from this list of "champions"
    final_selection = top_patterns_df.head((numFsp))

    # Pick the one with highest support
    if not final_selection.empty:

        for fsp in final_selection['Patterns']:
            fpgrowthFsp = set(map(int, fsp.split()))
            fspList.append(fpgrowthFsp)

            # Report the results
        print(f"Itemset: {[a for a in fspList]}")
        print(f"Moderate itemset size: {[len(a) for a in fspList]}")
        print("Support (minsup):", [f"{s:.3f}" for s in final_selection['Support']])
    
    else:
        print("No patterns found.")

    # Performance info
    runtimePami = obj.getRuntime()
    rssPami = obj.getMemoryRSS()

    print(f"Runtime (seconds): {runtimePami:.3f}")
    print(f"Memory usage (Bytes): {rssPami:.3f}")

    # Remove the transaction file
    os.remove(tempFile)

    return fspList, runtimePami, rssPami
    
# ___pami max fpgrowth algorithm___
def pamiMaxFpgrowth(aerData, numFsp, minSup):
    """
    Extract maximal frequent neuron activation patterns using the PAMI MaxFPGrowth algorithm.

    Args:
        aerData: AER spike event list containing (timestep, sample, neuron) tuples.
        numFsp: Number of representative activation patterns to return.
        minSup: Minimum support threshold required for frequent patterns.

    Returns:
        tuple:
            fspList: List of extracted maximal neuron activation patterns.
            runtimePami: Runtime reported by the PAMI MaxFPGrowth algorithm in seconds.
            rssPami: Memory usage reported by the PAMI algorithm in bytes.
    """

    # Initialize FSP (Frequent Spike Pattern)
    fspList = []
    
    # Group neuron spikes by timestep, sample into transactions
    transactionDict = defaultdict(list)
    for timestep, sample, neuron in aerData:
        transactionDict[(timestep, sample)].append(neuron)

    transactions = list(transactionDict.values())
    
    print(f"Total transactions: {len(transactions)}")

    # Temp file storing trasaction data
    tempFile = "transactionsTemp.txt"

    # Save transactions to file (PAMI input format)
    with open(tempFile, "w") as f:
        for txn in transactions:
            f.write(" ".join(map(str, txn)) + "\n")

    print("Running PAMI MaxFPGrowth()...")
    obj = PamiMaxFPGrowth.MaxFPGrowth(tempFile, minSup = minSup, sep = " ")
    obj.mine()

    # Get results as DataFrame
    patternsDf = obj.getPatternsAsDataFrame()  # columns: ['Patterns', 'Support']

    # Calculate pattern size
    patternsDf["length"] = (patternsDf["Patterns"].str.split().str.len())

    # Keep highest-support pattern for each pattern size
    final_selection = (patternsDf.sort_values(["length", "Support"], ascending=[False, False]).groupby("length").head(1).head(numFsp))
    
    # Pick the one with highest support
    if not final_selection.empty:

        for fsp in final_selection['Patterns']:
            fpgrowthFsp = set(map(int, fsp.split()))
            fspList.append(fpgrowthFsp)

            # Report the results
        print(f"Itemset: {[a for a in fspList]}")
        print(f"Moderate itemset size: {[len(a) for a in fspList]}")
        print("Support (minsup):", [f"{s:.3f}" for s in final_selection['Support']])
    
    else:
        print("No patterns found.")

    # Performance info
    runtimePami = obj.getRuntime()
    rssPami = obj.getMemoryRSS()

    print(f"Runtime (seconds): {runtimePami:.3f}")
    print(f"Memory usage (Bytes): {rssPami:.3f}")

    # Remove the transaction file
    os.remove(tempFile)

    return fspList, runtimePami, rssPami

# Compress a given neuron pattern or a part of it into a symbol
def compress(spikeData, patternMiningFunc, numFsp, pmt, num_hidden, minSup):
    """
    Compress spike data by replacing frequent neuron patterns with symbols.

    Args:
        spikeData: Spike tensor data.
        patternMiningFunc: Function used to mine frequent spike patterns.
        numFsp: Number of frequent spike patterns.
        pmt: Pattern matching threshold.
        num_hidden: Number of hidden neurons.

    Returns:
        tuple:
            Compressed AER data, original spike shape, pattern list,
            symbol list, and compression metrics.
    """

    aerData = convertToAER(spikeData)
    aerDataSizeLen = len(aerData)

    print(f"FSP Count: {numFsp}")
    print(f"PMT Value: {pmt}")

    PATTERNLIST, runTime, rss = patternMiningFunc(aerData, numFsp, minSup)
    baseSymbol = num_hidden
    symbolList = []

    print("Compressing the AER data with given patterns...")
    
    timestepDictInitial = {}

    compressData = []

    # Store both neuron list and neuron set to avoid repeated set creation
    for timestep, sample, neuron in aerData:

        if (timestep, sample) not in timestepDictInitial:
            timestepDictInitial[(timestep, sample)] = {
                "neurons": [],
                "set": set()
            }

        timestepDictInitial[(timestep, sample)]["neurons"].append(neuron)
        timestepDictInitial[(timestep, sample)]["set"].add(neuron)

    finalCompressedDict = defaultdict(list)

    # Precompute PMT thresholds
    patternThresholds = [math.ceil(len(PATTERN) * pmt) for PATTERN in PATTERNLIST]

    for idx, PATTERN in enumerate(PATTERNLIST):

        SYMBOL = baseSymbol + idx
        symbolList.append(SYMBOL)

        updatedTimestepDict = {}

        threshold = patternThresholds[idx]

        for (timestep, sample), data in timestepDictInitial.items():
            neurons = data["neurons"]
            neuronSet = data["set"]
            matched = PATTERN.intersection(neuronSet)

            if len(matched) >= threshold:

                # Add the compressed symbol
                finalCompressedDict[(timestep, sample)].append(SYMBOL)

                # Add all other neurons except the ones being compressed
                remainingSet = neuronSet - PATTERN

                # Update the list with only remaining neurons for next patterns
                if remainingSet:
                    updatedTimestepDict[(timestep, sample)] = {"neurons": list(remainingSet), "set": remainingSet}

            else:
                # Pattern doesn't match enough, keep the full list
                updatedTimestepDict[(timestep, sample)] = data


        # Update the timestepDict for next pattern iteration
        timestepDictInitial = updatedTimestepDict

    # Add any remaining neurons that weren't matched by any pattern
    for (timestep, sample), data in timestepDictInitial.items():

        finalCompressedDict[(timestep, sample)].extend(data["neurons"])

    for (timestep, sample), neurons in finalCompressedDict.items():

        for n in neurons:
            compressData.append((timestep, sample, n))

    compressDataSizeLen = len(compressData)

    if aerDataSizeLen > 0:
        reductionLen = ((aerDataSizeLen - compressDataSizeLen) * 100) / aerDataSizeLen

    else:
        aerDataSizeLen = 0
        reductionLen = 0

    print("Compressed AER data with given pattern...")
    print(f"Reduction with len(): {reductionLen:.2f}%")

    metrics = {
        "reduction_len": reductionLen,
        "runtime": runTime,
        "rss_mem": rss
    }

    return compressData, spikeData.shape, PATTERNLIST, symbolList, metrics

# Function to decompress the symbol back to the original neuron pattern
def decompress(net, aerData, PATTERNLIST, SYMBOLIST, shape):
    """
    Decompress symbol-based AER data back into spike tensor format.

    Args:
        net: snnTorch model used to determine the target computation device.
        aerData: Compressed AER spike data.
        PATTERNLIST: List of neuron patterns.
        SYMBOLIST: List of symbols mapped to patterns.
        shape: Original spike tensor shape.

    Returns:
        torch.Tensor: Reconstructed spike tensor.
    """

    print("Decompressing AER data...")
    decompressData = []
    symbolToPattern = dict(zip(SYMBOLIST, PATTERNLIST))

    # Estimate output size to reduce list resizing
    decompressData = []

    for timestep, sample, neuron in aerData:
        if neuron in symbolToPattern:
            # Add the neuron pattern
            decompressData.extend([(timestep, sample, p_neuron) for p_neuron in symbolToPattern[neuron]])

        else:
            # No compression; keep all neurons
            decompressData.append((timestep, sample, neuron))


    spikeData = convertToSpike(net, decompressData, shape)

    print("Decompressed AER data...\n")

    return spikeData
