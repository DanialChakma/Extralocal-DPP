# -*- coding: utf-8 -*-
"""
###############################################################################

The module is used for computing the composition of amino acids, dipetide and

3-mers (tri-peptide) for a given protein sequence. You can get 8420 descriptors

for a given protein sequence. You can freely use and distribute it. If you hava

any problem, you could contact with us timely!


Authors: Danial Chakma

Date: 2012.3.27

Email: danial08cse@gmail.com

###############################################################################
"""

import re
import math

AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
__AA_LETTERS__ = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
__AA_PAIRS__ = [A1 + A2 for A1 in __AA_LETTERS__ for A2 in __AA_LETTERS__]
__TRI_GRAMS__ = [ PAIR+AA for PAIR in __AA_PAIRS__ for AA in __AA_LETTERS__ ]
#############################################################################################
def CalculateAAComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of Amino acids

    for a given protein sequence.

    Usage:

    result=CalculateAAComposition(protein)

    Input: protein is a pure protein sequence.

    Output: result is a dict form containing the composition of

    20 amino acids.
    ########################################################################
    """
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result


#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of dipeptidefor a given protein sequence.

    Usage:

    result=CalculateDipeptideComposition(protein)

    Input: protein is a pure protein sequence.

    Output: result is a dict form containing the composition of

    400 dipeptides.
    ########################################################################
    """
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round(float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1) * 100, 2)
    return Result


#############################################################################################

def Getkmers():
    """
    ########################################################################
    Get the amino acid list of 3-mers.

    Usage:

    result=Getkmers()

    Output: result is a list form containing 8000 tri-peptides.

    ########################################################################
    """
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


#############################################################################################
def GetSpectrumDict(proteinsequence):
    """
    ########################################################################
    Calcualte the spectrum descriptors of 3-mers for a given protein.

    Usage:

    result=GetSpectrumDict(protein)

    Input: protein is a pure protein sequence.

    Output: result is a dict form containing the composition values of 8000

    3-mers.
    ########################################################################
    """
    proteinsequence = ''.join([AA for AA in proteinsequence if AA in __AA_LETTERS__])
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence))
    return result


#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of AADs, dipeptide and 3-mers for a

    given protein sequence.

    Usage:

    result=CalculateAADipeptideComposition(protein)

    Input: protein is a pure protein sequence.

    Output: result is a dict form containing all composition values of

    AADs, dipeptide and 3-mers (8420).
    ########################################################################
    """
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))
    result.update(GetSpectrumDict(ProteinSequence))

    return result


#############################################################################################
def CalculateAAC_plus_DipC(ProteinSequence):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))
    return result


def nGappedDip(sequence, gap=5):
    sequence = ''.join([AA for AA in sequence if AA in __AA_LETTERS__])
    if gap < 1:
        print('Gap parameter must be at least one.')
        return
    result = {}
    sequence = sequence.strip()
    for g in range(1, gap + 1):
        OccurenceDict = {}
        for pair in __AA_PAIRS__:
            OccurenceDict[pair] = 0
        #sum = 0
        L = len(sequence)
        TI = L - g - 1 # Total Iteration
        for i in range(0,TI):
            # start to stop-1, stop is non-inclusive
            j = i + g + 1
            if i < L and j < L and sequence[i] in __AA_LETTERS__ and sequence[j] in __AA_LETTERS__:
                OccurenceDict[sequence[i] + sequence[j]] = OccurenceDict[sequence[i] + sequence[j]] + 1
            #sum = sum + 1
        for pair in __AA_PAIRS__:
            result[pair + '.gap' + str(g)] = round(float(OccurenceDict[pair] / L),7) if L != 0 else 0

    return result
def TriGram(ProteinSequence):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    Result = {}
    for AAA in __TRI_GRAMS__:
        #Result[AAA] = round(float(ProteinSequence.count(AAA)) / (LengthSequence-2), 5)
        Result[AAA] = round(float(ProteinSequence.count(AAA)) / (LengthSequence), 7)
    return Result
def BiGram(ProteinSequence):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    Result = {}
    for AA in __AA_PAIRS__:
        #Result[AA] = round(float(ProteinSequence.count(AA)) / (LengthSequence-1), 5)
        Result[AA] = round(float(ProteinSequence.count(AA)) / (LengthSequence), 7)
    return Result
def MonoGram(ProteinSequence):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    Result = {}
    for A in __AA_LETTERS__:
        Result[A] = round(float(ProteinSequence.count(A)) / LengthSequence, 7)
    return Result

def MonoGramPercentile(ProteinSequence, percent=50):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    fraction = float(percent / 100)
    RefinedLength = int(LengthSequence * fraction)
    Result = {}
    for i in __AA_LETTERS__:
        #Result[i + '.P' + str(percent)] = round(float(ProteinSequence[0:RefinedLength].count(i)) /RefinedLength, 5)
        Result[i + '.P' + str(percent)] = round(float(ProteinSequence[0:RefinedLength].count(i)) /RefinedLength, 7)
    return Result
def BigramPercentile(ProteinSequence, percent=50):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    LengthSequence = len(ProteinSequence)
    RefinedLength = math.floor(((LengthSequence * percent) / 100))
    Result = {}
    for pair in __AA_PAIRS__:
        #Result[pair + '.P' + str(percent)] = round( float(ProteinSequence[0:RefinedLength].count(pair)) / (RefinedLength-1), 5 )
        Result[pair + '.P' + str(percent)] = round( float(ProteinSequence[0:RefinedLength].count(pair)) / (RefinedLength), 7 )
    return Result
def K_NN(ProteinSequence,k=30):
    ProteinSequence = ''.join([AA for AA in ProteinSequence if AA in __AA_LETTERS__])
    __LENGTH__ = len(ProteinSequence)
    __NORMALIZED_TERM__ =  __LENGTH__
    Result = {}
    for AA_1st in __AA_LETTERS__:
        for AA_2nd in __AA_LETTERS__:
            for nn in range(1,k+1):
                Result[AA_1st+'-k'+str(nn)+'n-'+AA_2nd] = 0
    for AA_1st in __AA_LETTERS__:

        try:
            __fst_index__ = ProteinSequence.index(AA_1st)
            for AA_2nd in __AA_LETTERS__:

                try:
                    __temp_index__ = __fst_index__
                    for nn in range(1,k+1):
                        try:
                            __second_index__ = ProteinSequence.index(AA_2nd,__temp_index__+1)
                            __distance__ = __second_index__ - __fst_index__
                            #Result[AA_1st + '-k' + str(nn) + 'n-' + AA_2nd] = round(((__distance__/__NORMALIZED_TERM__)*100),2) if __NORMALIZED_TERM__ != 0 else 0
                            Result[AA_1st + '-k' + str(nn) + 'n-' + AA_2nd] = round(((__distance__/__NORMALIZED_TERM__)),7) if __NORMALIZED_TERM__ != 0 else 0
                            __temp_index__ = __second_index__
                        except Exception as Err:
                            pass
                            #print('Inner Except:',Err)
                except Exception as Err:
                    pass
                    #print('Outer Exception:',Err)
        except Exception as Err:
            pass
    return Result
def GetMonoGramPercentiles(ProteinSequence):
    __PERCENTS__ = [10,20,30,40,50,60,70,80,90,100]
    Results = {}
    for percent in __PERCENTS__:
        res = MonoGramPercentile(ProteinSequence,percent)
        Results.update(res)
    return Results
def GetBiGramPercentiles(ProteinSequence):
    __PERCENTS__ = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Results = {}
    for percent in __PERCENTS__:
        res = BigramPercentile(ProteinSequence, percent)
        Results.update(res)
    return Results
#############################################################
def PSF(sequence, first_residue_len=10):
    aaPairs = [aa1 + aa2 for aa1 in AALetter for aa2 in AALetter]
    first_residues = sequence[:first_residue_len]
    aa_result = {}

    for pos in range(0, len(first_residues)):
        for aa in AALetter:
            aa_result[aa + str(pos)] = 0
            # print(aa+str(pos))
    for pos in range(0, len(first_residues)):
        for aa in AALetter:
            if aa == first_residues[pos]:
                aa_result[aa + str(pos)] = 1
    for pos in range(0, len(first_residues) - 1):
        for aap in aaPairs:
            aa_result[aap + str(pos)] = 0
            # print(aap+str(pos))
    for pos in range(0, len(first_residues)):
        for aap in aaPairs:
            if aap == first_residues[pos:pos + 2]:
                aa_result[aap + str(pos)] = 1
    return aa_result

def __AC_STAS__(ProteinSequence):
    __results__ ={}
    for AC in __AA_LETTERS__:
        __results__[AC] = ProteinSequence.count(AC)
    return __results__
if __name__ == "__main__":
    protein = "ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
    protein = "KKEKSX12PKGKSSISPQARAFLEEVFRRKQSLNSKEKEEVAKKCGITPLQVRVWFINKRMRSK"
    res = MonoGram(protein)
    print('Mongram:',res)
    res = BiGram(protein)
    print('Bigram:',res)
    res = TriGram(protein)
    print('Trigram:',res)
    res = nGappedDip(protein, gap=20)
    print('nGap', res)
    res = GetMonoGramPercentiles(protein)
    print('MonogramPercentiles:', res)
    res = GetMonoGramPercentiles(protein)
    print('BigramPercentiles:', res)

    res = K_NN(protein,k=5)
    print('KNN ',res)



