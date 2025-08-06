# %matplotlib qt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import pyplot as plt
import torch
from Utils.utils import norm


def plot_seg_results_paper(config, model_output_seg, hf_gt_seg_dice, lf_gt_seg_dice, prior_seg_dice):
    size = config["size"]
    size_lf = config["size_lf"]

    titles = ("WM-GT", "GM-GT", "CSF-GT","BG-GT", 
    "WM-Prediction", "GM-Prediction", "CSF-Prediction","BG-Prediction", 
    "WM-GT (ULF)", "GM-GT (ULF)", "CSF-GT (ULF)","BG-GT (ULF)", 
    "WM-Prior", "GM-Prior", "CSF-Prior", "BG-Prior"
    )

    fig, axes = plt.subplots(4,4, layout='constrained')
    
    im = im5 = axes[0,0].imshow(hf_gt_seg_dice[0,1,:,:].cpu().detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im6 = axes[0,1].imshow(hf_gt_seg_dice[0,2,:,:].cpu().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im7 = axes[0,2].imshow(hf_gt_seg_dice[0,3,:,:].cpu().detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im8 = axes[0,3].imshow(hf_gt_seg_dice[0,0,:,:].cpu().detach().numpy(), cmap = 'gray',vmin=0,vmax=1)


    im = im1 = axes[1,0].imshow(model_output_seg[:,:,0].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im2 = axes[1,1].imshow(model_output_seg[:,:,1].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im3 = axes[1,2].imshow(model_output_seg[:,:,2].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im4 = axes[1,3].imshow(model_output_seg[:,:,3].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)



    im = im13 = axes[2,0].imshow(lf_gt_seg_dice[0,1,:,:].cpu().view(size_lf).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im14 = axes[2,1].imshow(lf_gt_seg_dice[0,2,:,:].cpu().view(size_lf).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im15 = axes[2,2].imshow(lf_gt_seg_dice[0,3,:,:].cpu().view(size_lf).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    im = im16 = axes[2,3].imshow(lf_gt_seg_dice[0,0,:,:].cpu().view(size_lf).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)

    imx = im9 = axes[3,0].imshow(prior_seg_dice[0,1,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    imx = im10 = axes[3,1].imshow(prior_seg_dice[0,2,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    imx = im11 = axes[3,2].imshow(prior_seg_dice[0,3,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    imx = im12 = axes[3,3].imshow(prior_seg_dice[0,0,:,:].cpu().view(size).detach().numpy(), cmap='gray')

    for i in range(0,4):
        for j in range(0,4):
            axes[i,j].set_axis_off()
            
            
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=2 / 72, hspace=0, wspace=-2)    

    cbar = fig.colorbar(im, ax=axes[:3,:] ,shrink = 0.9, aspect=55*0.9)
    cbar2 = fig.colorbar(imx, ax=axes[3:,:], shrink =0.9)
    axes[0,0].set_title("White Matter", color='blue', fontsize = 8)
    axes[0,1].set_title("Gray Matter", color='blue', fontsize = 8)
    axes[0,2].set_title("CSF", color='blue', fontsize = 8)
    axes[0,3].set_title("Background", color='blue', fontsize = 8)
    
    
    # axes[0,0].set_title("GT", color='blue', fontsize = 9, rotation='vertical', x = -0.05, y =0.5)
    axes[0,0].annotate('Ground Truth', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 8, color='blue')
    axes[1,0].annotate('INR Prediction', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 8, color='red')
    axes[2,0].annotate('ULF Ground Truth', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 8, color='blue')
    axes[3,0].annotate('Prior', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 8, color='blue')
    # fig.tight_layout()
    return fig



def plot_final_results_paper(config, model_output_seg, model_output_img, hf_ground_truth, lf_gt, lf_output):
    size = config["size"]
    size_lf = config["size_lf"]
    hf_result = (model_output_img[:,:,0] * model_output_seg[:,:,0]) + (model_output_img[:,:,1]* model_output_seg[:,:,1]) + (model_output_img[:,:,2] * model_output_seg[:,:,2]) + (model_output_img[:,:,3] * model_output_seg[:,:,3])
    
    titles = ("HF-GT", "HF-Prediction", "LF Synthetic GT", "LF-Prediction")

    fig, axes = plt.subplots(1,4, layout='constrained')

    im = im1 = axes[0].imshow(norm(hf_ground_truth), cmap = 'gray')
    im = im2 = axes[1].imshow(norm(hf_result.flatten().view(size).detach().cpu()), cmap='gray')
    im = im3 = axes[2].imshow(norm(lf_gt.view(size_lf)), cmap = 'gray')
    im = im4 = axes[3].imshow(norm(lf_output.flatten().view(size_lf).detach().cpu()), cmap = 'gray')
 

    axes[0].set_title(titles[0],color="blue", fontsize=9)
    axes[1].set_title(titles[1],color="red", fontsize=9)
    axes[2].set_title(titles[2],color="blue", fontsize=9)
    axes[3].set_title(titles[3],color="blue", fontsize=9)
    
    
    for i in range(0,4):
        axes[i].set_axis_off()
            
            
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=2 / 72, hspace=0, wspace=0)    
    cbar = fig.colorbar(im, ax=axes ,shrink = 0.27, aspect=20*0.9,pad=0.02)
    
    return fig


def plot_hf_results_paper(config, model_output_seg, model_output_img, hf_ground_truth, lf_gt, lf_output , interpolated_bilinear, interpolated_bicubic):
    size = config["size"]
    size_lf = config["size_lf"]
    hf_result = (model_output_img[:,:,0] * model_output_seg[:,:,0]) + (model_output_img[:,:,1]* model_output_seg[:,:,1]) + (model_output_img[:,:,2] * model_output_seg[:,:,2]) + (model_output_img[:,:,3] * model_output_seg[:,:,3])
    titles = ("HF-GT", "INR Prediction", "Bilinear Interpolated","Bicubic Interpolated")

    fig, axes = plt.subplots(1,4, layout='constrained')

    im = im5 = axes[0].imshow(norm(hf_ground_truth), cmap='gray')
    im = im6 = axes[1].imshow(norm(hf_result.flatten().view(size).detach().cpu()), cmap='gray')
    im = im7 = axes[2].imshow(norm(interpolated_bilinear.view(size)), cmap='gray')
    im = im8 = axes[3].imshow(norm(interpolated_bicubic.view(size)), cmap='gray')

    axes[0].set_title(titles[0],color="blue", fontsize=9)
    axes[1].set_title(titles[1],color="red", fontsize=9)
    axes[2].set_title(titles[2],color="blue", fontsize=9)
    axes[3].set_title(titles[3],color="blue", fontsize=9)

    for i in range(0,4):
        axes[i].set_axis_off()
                
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=2 / 72, hspace=0, wspace=0)    
    cbar = fig.colorbar(im, ax=axes ,shrink = 0.27, aspect=20*0.9,pad=0.02)
    return fig



def plot_sensitivity_results(images_dict, col_titles, bottom_titles, subplot_labels=None, cmap='gray'):
    """
    #Credits : Used ChatGPT to modify certain parts of this function
    Plot a 3x7 grid of images stored in a dictionary with row titles as keys.

    Parameters:
    - images_dict: dict of image lists with shape [3][7] e.g., {"GT": [...], "Pred": [...], ...}
    - col_titles: list of 7 column titles
    - subplot_labels: optional dict matching the same structure as images_dict
    - cmap: color map for imshow
    """
    row_titles = list(images_dict.keys())
    n_rows = len(row_titles)
    n_cols = len(col_titles)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.2 * n_rows))

    for i, row_key in enumerate(row_titles):
        for j in range(n_cols):
            ax = axs[i][j]
            ax.imshow(images_dict[row_key][j].detach().cpu().T, cmap=cmap)
            ax.axis('off')
            if subplot_labels:
                ax.set_title(subplot_labels[row_key][j], fontsize='x-small')

    # Column titles
    for j in range(n_cols):
        axs[0][j].set_title(col_titles[j], fontsize='medium', pad=10)

    # Row titles (Y-axis)
    row_y_positions = [0.85, 0.55, 0.25][:n_rows]  # Adjust if n_rows != 3
    for i, row_title in enumerate(row_titles):
        c ='b' if i<2 else 'r'
        fig.text(0.045, row_y_positions[i], row_title, va='center', ha='right',
                 fontsize='medium', rotation=90, color = c)
    #Bottom titles 
    for j in range(n_cols):
        gap=0.75 if j<2 else 0.75
        fig.text(
            0.1 + j * (gap / (n_cols - 1)),  # adjust horizontal placement
            0.085,  # vertical position
            bottom_titles[j],
            ha='center',
            va='top',
            fontsize='small'
        )

    [axs_inv.invert_yaxis() for axs_inv in axs.flatten()]
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.05, top=0.99)

    plt.subplots_adjust(
    left=0.05, right=0.9, top=0.99, #bottom=0.12,
    wspace=-0.001,  # Horizontal space between subplots
    hspace=0.001   # Vertical space between subplots
    )

    plt.show()
    return fig





