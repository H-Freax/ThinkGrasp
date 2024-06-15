import numpy as np
import os


root = '../grasp_data'

for i in range(87):
    eg = np.load(os.path.join(root, 'new_score', '{}_labels.npz'.format(str(i).zfill(3))))
    s5 = np.load(os.path.join(root, 'score5', '{}_labels.npz'.format(str(i).zfill(3))))
    print(eg.files)
    mask_idx = eg['mask_idx']
    score2 = eg['score2']
    score3 = eg['score3']
    score4 = eg['score4']
    score5 = s5['score5']

    score2_norm = 1 - (score2-np.min(score2))/(np.max(score2)-np.min(score2))

    obj_path = os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3)))
    label = np.load(obj_path)
    label_score = label['scores']
    mask1 = (label_score > 0)
    fric_coefs = label_score[mask1]
    scores = 1.1 - fric_coefs

    idx = np.array([220872, 434776, 213653, 8151, 222708])

    final_score = 0.7*scores+0.05*score2_norm+0.2*score3*score4+0.05*score5
    final_score_norm = (final_score-np.min(final_score))/(np.max(final_score)-np.min(final_score))
    # idx = np.array([220872, 434776, 213653, 8151, 222708])
    print(score2_norm[idx])
    print((score3 * score4)[idx])
    # print(0.05*score5[idx])
    # print(final_score[idx])
    # print(final_score_norm[idx])
    label_score[mask1] = final_score_norm
    savepath = os.path.join(root, 'new_grasp_label_2345', '{}_labels.npz'.format(str(i).zfill(3)))
    np.savez(savepath, new_scores=label_score)

print('1')