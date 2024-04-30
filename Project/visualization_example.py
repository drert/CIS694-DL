import os
from scipy.io import loadmat
import numpy as np

from tqdm import tqdm # progress bar visualizer
import torch, torchvision
from torchvision.io import read_image
import copy 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from adjustText import adjust_text

MAX_IMGs = 100





mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                   12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

mpii_template = dict([(mpii_idx_to_jnt[i], []) for i in range(16)])

path = "C:/Users/Drert/Documents/MPII"
img_path = path + "/Images/images/"
annot_file = "/Annotations/mpii_human_pose_v1_u12_1.mat"
test_image = "000001163.jpg"

ml_mpii = loadmat(path+annot_file, struct_as_record=False)['RELEASE'][0,0]
num_images = annotation_mpii = ml_mpii.__dict__['annolist'][0].shape[0]

print(num_images)



def visualize_image(image_info):
    '''
    :param image_info: (dict)
    '''
    colour = {'rankl': (0, 0, 1), 'rknee': (0, 0, 1), 'rhip': (0, 0, 1),
              'lankl': (1, 0, 0), 'lknee': (1, 0, 0), 'lhip': (1, 0, 0),
              'rwri': (1, 1, 0), 'relb': (1, 1, 0), 'rsho': (1, 1, 0),
              'lwri': (0, 1, 0), 'lelb': (0, 1, 0), 'lsho': (0, 1, 0),
              'head': (0, 1, 1), 'thorax': (0, 1, 1), 'upper_neck': (0, 1, 1)}

    os.makedirs(os.path.join(path, 'results', 'viz_gt'), exist_ok=True)
    img_dump =  os.path.join(path, 'results', 'viz_gt')

    # Since we're considering only MPII, the outer loop will execute only once.
    for dataset_name_ in image_info.keys():
        # Iterate over all images
        for i in tqdm(range(len(image_info[dataset_name_]['img']))):

            fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False)
            ax.set_axis_off()

            # Load image, gt for the given index
            img = image_info[dataset_name_]['img'][i]
            img_name = image_info[dataset_name_]['img_name'][i]
            img_gt = image_info[dataset_name_]['img_gt'][i]

            # Store joint names which will be displayed on the image
            text_overlay = []
            ax.imshow(img)

            # Color-code the joint and joint name onto the image
            joint_names = list(colour.keys())
            for jnt in joint_names:
                for jnt_gt in img_gt[jnt]:
                    if jnt_gt[2]:
                        text_overlay.append(ax.text(x=jnt_gt[0], y=jnt_gt[1], s=jnt, color=colour[jnt], fontsize=6))
                        ax.add_patch(Circle(jnt_gt[:2], radius=1.5, color=colour[jnt], fill=False))

            # Ensure no crowding of joints on the image
            adjust_text(text_overlay)

            plt.savefig(fname=os.path.join(img_dump, '{}'.format(img_name)),
                        facecolor='black', edgecolor='black', bbox_inches='tight', dpi=300)

            plt.close()
            del fig, ax













init_idx = 0
batch_size = 10

while init_idx < num_images and init_idx < MAX_IMGs :
    img_dict = {"mpii": {"img" : [], "img_name" : [], "img_pred" : [], "img_gt" : []}}
    
    for idx in tqdm(range(init_idx, min(init_idx+batch_size, num_images))) :
        annots = ml_mpii.__dict__["annolist"][0,idx]
        train  = ml_mpii.__dict__["img_train"][0,idx].flatten()[0]
        p_id   = ml_mpii.__dict__["single_person"][idx][0].flatten()
        print(train)
        img_name = annots.__dict__["image"][0, 0].__dict__["name"][0]
        try :
            image = plt.imread(img_path + img_name)
        except FileNotFoundError :
            print("Failed to load {}".format(img_name))
            continue

        gt = copy.deepcopy(mpii_template)
        
        annotated = False

        for person in (p_id - 1) :
            try :
                annopoints = annots.__dict__["annorect"][0,person].__dict__["annopoints"][0,0]
                num_joints = annopoints.__dict__["point"][0].shape[0]

                for i in range(num_joints) :
                    joint = annopoints.__dict__["point"][0,i]

                    x = joint.__dict__["x"].flatten()[0]
                    y = joint.__dict__["y"].flatten()[0]

                    id_ = joint.__dict__["id"][0][0]
                    vis = joint.__dict__["is_visible"].flatten()

                    if vis.size == 0:
                        vis = 1
                    else :
                        vis = vis.item()
                    
                    gt_joint = np.array([x,y,vis]).astype(np.float16)
                    gt[mpii_idx_to_jnt[id_]].append(gt_joint)

                annotated = True
            except KeyError :
                continue
        if not annotated : 
            continue

        img_dict['mpii']['img'].append(image)
        img_dict['mpii']['img_name'].append(img_name)
        img_dict['mpii']['img_gt'].append(gt)

    # visualize_image(img_dict)
    init_idx += batch_size


















