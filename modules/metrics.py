from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from RaTEScore.RaTEScore.scorer import *
from RaTEScore.RaTEScore.score import *


# We run these 2 seperalty after training (in different OS and env) as there was a bug in the train envrionment.


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        # (RaTEScore(), "RaTEScore")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            if method=='RaTEScore':
                gts = [item[0] for item in gts.values()]
                res = [item[0] for item in res.values()]

                score = scorer.compute_score(res, gts)
                score = sum(score) / len(score)
            else:
                score, scores = scorer.compute_score(gts, res, verbose=0)
        except:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
