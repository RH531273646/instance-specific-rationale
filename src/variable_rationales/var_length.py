import torch
from torch import nn
import math 
import json
import numpy as np
from src.common_code.metrics import jsd, kl_div_loss, perplexity, simple_diff

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from src.common_code.useful_functions import mask_topk, mask_contigious
import math

nn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)


div_funs = {
    "jsd":jsd, 
    "perplexity":perplexity, 
    "kldiv":kl_div_loss,
    "classdiff": simple_diff
}


def rationale_length_computer_(
    model, inputs, scores, y_original, 
    results_dict, feature_attribution, 
    zero_logits, original_sents, 
    fidelity = "lower_fidelity"):


    divergence_fun = div_funs[args.divergence]

    """
    function to calculate for a batch:
        * variable rationale length
        * variable rationale mask
        * fixed rationale mask
    for a specific set of importance scores
    """

    assert fidelity in ["max_fidelity", "lower_fidelity"]

    max_rationale = args.rationale_length 

    tokens = max_rationale * inputs["lengths"].float().mean()

    ## if we break down our search in increments
    if fidelity == "lower_fidelity":
        
        per_how_many = 2/100 ## skip every per_how_many percent of tokens
        percent_to_tokens = round(per_how_many * int(min(inputs["lengths"]))) ## translate percentage to tokens

        ## special case for very short sequences in SST and AG of less than 6 tokens
        if percent_to_tokens == 0:
            
            collector = torch.zeros([math.ceil(tokens), original_sents.size(0)])

            grange = range(0, math.ceil(tokens))
        
        ## for longer than 4 word sequences
        else:

            grange = range(percent_to_tokens, math.ceil(tokens) + percent_to_tokens, percent_to_tokens) ## convert to range with increments

            ## empty matrix to collect scores 
            ## // +1 to keep empty first column like below (0 token)
            collector = torch.zeros([len(grange), original_sents.size(0)])
        
    ## else if we consider and search on every token
    else:
        
        collector = torch.zeros([math.ceil(tokens), original_sents.size(0)])

        grange = range(0, math.ceil(tokens))

    model.eval()
    stepwise_preds = []
    ## begin search
    with torch.no_grad():
        
        for j, _tok in enumerate(grange):

            if j == 0: _tok = 1
            
            if args.thresholder == "topk":

                inputs["sentences"] = mask_topk(original_sents, scores, _tok)

            else:

                inputs["sentences"] = mask_contigious(original_sents, scores, _tok)

            yhat, _ = model(**inputs)

            stepwise_preds.append(yhat.argmax(-1).detach().cpu().numpy())

            ### normalized divergence
            full_div = divergence_fun(
                torch.softmax(y_original - zero_logits, dim = -1), 
                torch.softmax(yhat - zero_logits, dim = -1)
            ) 

            collector[j] = full_div.detach().cpu()

    #### in short sequences (e.g. where grange is 0) it means they are formed from one token
    #### so that token is our explanation

    stepwise_preds = np.stack(stepwise_preds).T

    assert stepwise_preds.shape[0] == y_original.size(0)

    max_div, indxes = collector.max(0)
    indxes[indxes == 0] = 1


    ## now to generate the rationale ratio
    ## and other data that we care about saving
    for _i_ in range(y_original.size(0)):

        annot_id = inputs["annotation_id"][_i_]
        
        full_text_length = inputs["lengths"][_i_]
        rationale_length = indxes[_i_].detach().cpu().item()
        rationale_ratio = rationale_length / (full_text_length.float().detach().cpu().item() - 2)

        ## now to create the mask of variable rationales
        ## rationale selected (with 1's)
        if args.thresholder == "topk":

            rationale_mask = (mask_topk(original_sents[_i_], scores[_i_],rationale_length) == 0).long().detach().cpu().numpy()

        else:

            rationale_mask = (mask_contigious(original_sents[_i_], scores[_i_],rationale_length) == 0).long().detach().cpu().numpy()

        ## now to create the mask of variable rationales
        ## rationale selected (with 1's)

        fixed_rationale_length = math.ceil(args.rationale_length * inputs["lengths"][_i_].float())

        if args.thresholder == "topk":

            fixed_rationale_mask = (mask_topk(original_sents[_i_], scores[_i_], fixed_rationale_length) == 0).long().detach().cpu().numpy()

        else:

            fixed_rationale_mask = (mask_contigious(original_sents[_i_], scores[_i_],fixed_rationale_length) == 0).long().detach().cpu().numpy()

        ## we also need the fixed max div for selecting through the best rationales
        fixed_div = collector[-1,:][_i_] ## the last row of divergences represents the max rationale length

        results_dict[annot_id][feature_attribution] = {
            "variable rationale length" : rationale_length,
            "fixed rationale length" : fixed_rationale_length,
            "variable rationale ratio" : rationale_ratio, 
            "variable rationale mask" : rationale_mask,
            "fixed rationale mask" : fixed_rationale_mask,
            f"fixed-length divergence" : fixed_div.cpu().item(),
            f"variable-length divergence" : max_div[_i_].cpu().item(),
            "importance scores" : scores[_i_].cpu().detach().numpy(),
            "running predictions" : stepwise_preds[_i_]
        }

    return 

import os 
from tqdm import trange

def get_rationale_metadata_(model, data_split_name, data, model_random_seed):

    desc = f'creating rationale data for {data_split_name}'
    
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    
    rationale_results = {}

    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = [torch.stack(t).transpose(0,1) if type(t) is list else t for t in batch]
        
        inputs = {
            "sentences" : batch[0].to(device),
            "lengths" : batch[1].to(device),
            "labels" : batch[2].to(device),
            "annotation_id" : batch[3],
            "query_mask" : batch[4].to(device),
            "token_type_ids" : batch[5].to(device),
            "attention_mask" : batch[6].to(device),
            "retain_gradient" : True
        }
                                        
        assert inputs["sentences"].size(0) == len(inputs["labels"]), "Error: batch size for item 1 not in correct position"
        
        original_prediction, attentions =  model(**inputs)

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        #embedding gradients
        embed_grad = model.wrapper.model.embeddings.word_embeddings.weight.grad
        g = embed_grad[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        # cutting to length to save time
        attentions = attentions[:,:max(inputs["lengths"])]
        query_mask = inputs["query_mask"][:,:max(inputs["lengths"])]

        em = model.wrapper.model.embeddings.word_embeddings.weight[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        gradients = (g* em).sum(-1).abs() * query_mask.float()

        integrated_grads = model.integrated_grads(
                original_grad = g, 
                original_pred = original_prediction.max(-1),
                **inputs    
        )

        normalised_random = torch.randn(attentions.shape).to(device)

        normalised_random = torch.masked_fill(normalised_random, ~query_mask.bool(), float("-inf"))
        normalised_random = torch.softmax(normalised_random, dim = -1)

        # normalised integrated gradients of input
        normalised_ig = torch.masked_fill(integrated_grads[:, :max(inputs["lengths"])], ~query_mask.bool(), float("-inf"))

        # normalised gradients of input
        # normalised_grads = model.normalise_scores(gradients, inputs["sentences"][:, :max(inputs["lengths"])])
        normalised_grads = torch.masked_fill(gradients[:, :max(inputs["lengths"])], ~query_mask.bool(), float("-inf"))

        # normalised attention
        # normalised_attentions = model.normalise_scores(attentions * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])
        normalised_attentions = torch.masked_fill(attentions[:, :max(inputs["lengths"])], ~query_mask.bool(), float("-inf"))

        # retrieving attention*attention_grad
        attention_gradients = model.weights_or.grad[:,:,0,:].mean(1)[:,:max(inputs["lengths"])]
        
        attention_gradients =  (attentions * attention_gradients)[:, :max(inputs["lengths"])]
        
        # softmaxing due to negative attention gradients 
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        normalised_attention_grads = torch.masked_fill(attention_gradients, ~query_mask.bool(), float("-inf"))

        for _i_ in range(attentions.size(0)):
            
            annotation_id = inputs["annotation_id"][_i_]
            
            ## setting up the placeholder for storing the  rationales
            rationale_results[annotation_id] = {}
            rationale_results[annotation_id]["original prediction"] = original_prediction[_i_].detach().cpu().numpy()
            rationale_results[annotation_id]["thresholder"] = args.thresholder
            rationale_results[annotation_id]["divergence metric"] = args.divergence

        to_save_time = {
            "random" : normalised_random,
            "attention" : normalised_attentions,
            "gradients" : normalised_grads,
            "ig" : normalised_ig,
            "scaled attention" : normalised_attention_grads
        }


        original_sents = inputs["sentences"].clone()

        inputs["sentences"] = original_sents * torch.zeros_like(original_sents).to(device)

        zero_logits, _ =  model(**inputs)
        
        ## percentage of flips
        for feat_name , feat_score in to_save_time.items():

            rationale_length_computer_(
                model = model, 
                inputs = inputs, 
                scores = feat_score, 
                y_original = original_prediction, 
                zero_logits = zero_logits,
                original_sents=original_sents,
                fidelity = "max_fidelity",
                feature_attribution = feat_name, 
                results_dict = rationale_results
            )

        ## select best fixed (fixed-len + var-feat) and variable rationales (var-len + var-feat) and save 
        for _i_ in range(attentions.size(0)):

            annotation_id = inputs["annotation_id"][_i_]

            ## initiators
            init_fixed_div = float("-inf")
            init_var_div = float("-inf")

            for feat_name in {"attention", "scaled attention", "gradients", "ig"}:
                
                fixed_div = rationale_results[annotation_id][feat_name]["fixed-length divergence"]
                var_div = rationale_results[annotation_id][feat_name]["variable-length divergence"]

                if fixed_div > init_fixed_div:

                    rationale_results[annotation_id]["fixed-len_var-feat"] = rationale_results[annotation_id][feat_name]
                    rationale_results[annotation_id]["fixed-len_var-feat"]["feature attribution name"] = feat_name

                    init_fixed_div = fixed_div


                if var_div > init_var_div:

                    rationale_results[annotation_id]["var-len_var-feat"] = rationale_results[annotation_id][feat_name]
                    rationale_results[annotation_id]["var-len_var-feat"]["feature attribution name"] = feat_name

                    init_var_div = var_div

        pbar.update(data.batch_size)

    ## save rationale masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        ""
    )

    os.makedirs(fname, exist_ok= True)

    print(f"saved -> {fname}")

    np.save(fname + data_split_name + "-rationale_metadata.npy", rationale_results)

    return

            