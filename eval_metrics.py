from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score

import evaluate
rouge = evaluate.load('rouge')


   
def bleu_all(ref, hyp, weights=[0.25, 0.25, 0.25, 0.25], epsilon=0.1, alpha=5, k=5):
    """ (list of token lists, list of tokens, int) -> float
    Calculate BLEU score between a hypothesis and a list of references.
    epsilon (float) : the epsilon value use in method 1
    alpha (int) : the alpha value use in method 6
    k (int) : the k value use in method 4
    """
    # if isinstance(ref[0], list):
    #     ref = [r for ref_list in ref for r in ref_list]
    smooth = SmoothingFunction()
    # no smoothing
    bleu0 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method0)  
    # Add epsilon counts to precision with 0 counts  
    bleu1 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method1)
    # Add 1 to both numerator and denominator    
    bleu2 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method2) 
    # NIST geometric sequence smoothing    
    bleu3 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method3) 
    # better for shorter sentences   
    bleu4 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method4)    
    # similar matched counts for similar values of n  
    bleu5 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method5) 
    # Interpolates the maximum likelihood estimate of the precision p_n with a prior estimate pi0   
    #bleu6 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method6)    
    bleu6 = 0  # BROKEN FOR SOME REASON
    # Interpolates methods 4 and 5
    bleu7 = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth.method7)    
    return bleu0, bleu1, bleu2, bleu3, bleu4, bleu5, bleu6, bleu7
    # return sentence_bleu(ref, hyp, weights=[1. / n] * n, smoothing_function=SmoothingFunction().method1)
    

    
# HELPER FUNCTIONS FOR BLEU
def bleu_sent(ref, hyp, smooth_method = -1):
    """ (list of tokens, list of tokens, int) -> float
    smooth_method: 0-7, -1 for max
    """
    if smooth_method == -1:
        bleu_scores = bleu_all(ref, hyp, weights=[0.25, 0.25, 0.25, 0.25], 
                               epsilon=0.1, alpha=5, k=5)
        max_bleu = max(bleu_scores)
        max_index = bleu_scores.index(max_bleu)
        return (max_bleu, max_index)
    else:
        return (bleu_all(ref, hyp)[smooth_method], smooth_method)



def eval_bleu(predictions, correct, smooth_method = 7):
    """ (list, list, int) -> float
    """
    if isinstance(predictions[0], str):
        predictions = [sent.split() for sent in predictions]
    if isinstance(correct[0], str):
        correct = [sent.split() for sent in correct]
    sum_bleu = 0
    for i in range(len(predictions)):
        bleu_s = bleu_sent([predictions[i]], correct[i], smooth_method)
        sum_bleu += bleu_s[0]
    return sum_bleu / len(predictions)

def eval_meteor(predictions, correct):
    """ (list, list) -> float
    """
    if isinstance(predictions[0], str):
        predictions = [sent.split() for sent in predictions]
    if isinstance(correct[0], str):
        correct = [sent.split() for sent in correct]
    sum_meteor = 0
    for i in range(len(predictions)):
        sum_meteor += meteor_score([predictions[i]], correct[i])
    return sum_meteor / len(predictions)

def eval_rouge(predictions, correct):
    if isinstance(predictions[0], list):
        predictions = [' '.join(tokens) for tokens in predictions]
    if isinstance(correct[0], list):
        references = [' '.join(tokens) for tokens in correct]
    else:
        references = correct
    return rouge.compute(predictions=predictions, references=references)

 
