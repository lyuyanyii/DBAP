import torch
import torch.utils.data as data
import numpy as np
import json
import os
from collections import OrderedDict

class BindingAffinityDataset( data.Dataset ):
    charProtSet = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
    "U": 19, "T": 20, "W": 21, 
    "V": 22, "Y": 23, "X": 24, 
    "Z": 25 }
    
    charProtLen = 25
    
    charCompSet = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
    
    charCompLen = 64

    def _labelProtein( self, protein ):
        proteinLabel = np.zeros( self.maxProteinLen ).astype(np.int32)
        for i, ch in enumerate(protein[:self.maxProteinLen]):
            proteinLabel[i] = self.charProtSet[ch]
        return proteinLabel

    def _labelCompound( self, compound ):
        compoundLabel = np.zeros( self.maxCompoundLen ).astype(np.int32)
        for i, ch in enumerate(compound[:self.maxCompoundLen]):
            compoundLabel[i] = self.charCompSet[ch]
        return compoundLabel

    def __init__( self, dataset, mode, maxProteinLen = 1200, maxCompoundLen=100, fashion="single" ):
        super().__init__()

        self.maxProteinLen = maxProteinLen
        self.maxCompoundLen = maxCompoundLen

        data_dir = "data/{}/".format( dataset )
        proteins = json.load( open( os.path.join(data_dir, "{}_proteins_seq.txt".format(dataset)), "r" ), object_pairs_hook = OrderedDict)
        compounds = json.load( open( os.path.join(data_dir, "{}_compound_smiles.txt".format(dataset)), "r"), object_pairs_hook = OrderedDict)

        np.random.seed(0)

        proteins = [self._labelProtein(item) for (key, item) in proteins.items()]
        compounds = [self._labelCompound(item) for (key, item) in compounds.items()]

        indexProteins = np.random.permutation( len(proteins) )
        indexCompounds = np.random.permutation( len(compounds) )
        lenProteins = len(proteins)
        lenCompounds = len(compounds)

        if fashion == 'single':
            if mode == "train":
                indexProteins = indexProteins[:int(lenProteins*0.8)]
                indexCompounds = indexCompounds[:int(lenCompounds*0.8)]
            elif mode == "test":
                if dataset == "davis":
                    ratio = 0.8
                else:
                    ratio = 0.9
                indexProteins = indexProteins[int(lenProteins*ratio):]
                indexCompounds = indexCompounds[int(lenCompounds*ratio):]

        affinityPairs = []
        affinityFile = open( os.path.join(data_dir, "{}_binding_affinity.txt".format(dataset)), "r")
        for compoundID, line in enumerate(affinityFile.readlines()):
            if compoundID not in indexCompounds:
                continue
            for proteinID, score in enumerate(line.split()):
                if proteinID not in indexProteins or score == "nan":
                    continue
                score = np.float32(score)
                if dataset == "davis":
                    score = -(np.log10(score/1e9))
                affinityPairs.append( (proteinID, compoundID, score) )

        self.proteins = proteins
        self.compounds = compounds
        self.affinityPairs = affinityPairs
        if fashion == "pair":
            np.random.shuffle( self.affinityPairs )
            if mode =="train":
                self.affinityPairs = self.affinityPairs[ :int(len(self.affinityPairs)-1200) ]
            else:
                self.affinityPairs = self.affinityPairs[ int(len(self.affinityPairs)-1200): ]
        self.len = len(self.affinityPairs)

    def __len__( self ):
        return self.len

    def __getitem__( self, index ):
        protein = self.proteins[ self.affinityPairs[index][0] ]
        compound = self.compounds[ self.affinityPairs[index][1] ]
        affinityScore = self.affinityPairs[index][2]

        return [torch.LongTensor(protein), torch.LongTensor(compound), torch.FloatTensor([affinityScore])]

if __name__ == '__main__':
    davisDataset = BindingAffinityDataset( "davis", "train" )
    print( len(davisDataset) )
    davisDataset = BindingAffinityDataset( "davis", "test" )
    print( len(davisDataset) )
