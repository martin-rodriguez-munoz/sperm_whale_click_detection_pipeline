import numpy as np
from scipy.ndimage import label as get_connected_components
import pandas as pd

def get_prediction_times(model_conf,pred_times,pred_th):
    df = pd.DataFrame()
    df["confidence"] = model_conf
    model_conf = df["confidence"]

    above_th = np.ones(len(model_conf))*(model_conf>pred_th)
    components, num_components = get_connected_components(above_th)
    pred_center = [(model_conf[components == i+1]).idxmax() for i in range(num_components)]
    #best_prob = [np.max(model_conf[components == i+1]) for i in range(num_components)]

    return np.array(pred_times)[pred_center]
    



def get_TP_FP_FN(pred_list, gt_list, time_diff_threshold = 0.0045351):
    """
    Returns the number of true positives, false positives and false negatives
    """
    TP = 0
    FP = 0
    FN = 0



    # Here we keep track of: The id of the closest prediction to each ground truth coda, the difference in start times and, the different in end times
    gt_closest_pred_start_dist_end_dist = -np.ones((len(gt_list),2))

    # Here we keep track of: The id of the closest ground truth to each prediction coda, the difference in start times and, the different in end times
    pred_closest_gt_start_dist_end_dist = -np.ones((len(pred_list),2))

    # In this loop we calculate these values
    # An ID of -1 just means it hasn't been initialized yet
    gt_id = 0
    for gt_click_time in gt_list:

        
        pred_id = 0
        for pred_click_time in pred_list:
            time_diff = np.abs(gt_click_time-pred_click_time)

            if  gt_closest_pred_start_dist_end_dist[gt_id][0] == -1 or time_diff < gt_closest_pred_start_dist_end_dist[gt_id][1]:
                gt_closest_pred_start_dist_end_dist[gt_id][0] = pred_id
                gt_closest_pred_start_dist_end_dist[gt_id][1] = time_diff

            if pred_closest_gt_start_dist_end_dist[pred_id][0] == -1 or time_diff < pred_closest_gt_start_dist_end_dist[pred_id][1]:
                pred_closest_gt_start_dist_end_dist[pred_id][0] = gt_id
                pred_closest_gt_start_dist_end_dist[pred_id][1] = time_diff

            pred_id += 1

        gt_id += 1


    # For every value in the ground truth if we found a match that's a true positive otherwise it's a false negative.
    for gt_id in range(len(gt_list)):
        if gt_closest_pred_start_dist_end_dist[gt_id][0] == -1:
            #print(gt_list[gt_id])
            FN += 1
            continue


        if pred_closest_gt_start_dist_end_dist[int(gt_closest_pred_start_dist_end_dist[gt_id][0])][0] == gt_id and gt_closest_pred_start_dist_end_dist[gt_id][1] < time_diff_threshold:
            TP += 1
        else:
            #print(gt_list[gt_id])
            FN += 1

    # For every value in the prediction if we found a match that's a true positive otherwise it's a false positive.
    for pred_id in range(len(pred_list)):
        if pred_closest_gt_start_dist_end_dist[pred_id][0] == -1:
            FP += 1
            continue

        if gt_closest_pred_start_dist_end_dist[int(pred_closest_gt_start_dist_end_dist[pred_id][0])][0] == pred_id and pred_closest_gt_start_dist_end_dist[pred_id][1] < time_diff_threshold:
            continue
            # True positives have already been counted in the previous loop, don't count them again.
        else:
            FP += 1
    

    return TP, FP, FN