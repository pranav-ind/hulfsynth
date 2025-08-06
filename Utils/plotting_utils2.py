# %matplotlib qt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import pyplot as plt
import torch
from Utils.utils import norm, get_full_img


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





import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


def annotate_scores(ax, psnr, ssim, fontsize=6):
    ax.text(
        0.5, -0.08,  # x=centered, y=slightly below the image
        f"PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=fontsize,
        color='black'
    )

def plot_final_results_compare_paper(config, model_output_seg_list, prior_seg_dice, hf_ground_truth_list, model_output_img_list, lf_gt, lf_output, interpolated_bicubic, interpolated_bilinear, df):
    size = config["size"]
    size_lf = config["size_lf"]

    fig = plt.figure(figsize=(8, 5))
    
    # 4 rows: rows 0,1,3 for images; row 2 for vertical spacing
    gs = gridspec.GridSpec(4, 4,
                           height_ratios=[1, 1, 0.15, 1],
                           hspace=0, wspace=0,
                           top=1, bottom=0, left=0, right=1)

    # Row 0 (first row of HF images)
    ax00 = fig.add_subplot(gs[0, 0]); ax00.imshow(norm(hf_ground_truth_list[0]).T, cmap='gray', vmin=0, vmax=1)
    ax01 = fig.add_subplot(gs[0, 1]); ax01.imshow(norm(get_full_img(model_output_seg_list[0], model_output_img_list[0], config)).cpu().view(size).detach().numpy().T, cmap='gray')
    ax02 = fig.add_subplot(gs[0, 2]); ax02.imshow(norm(interpolated_bilinear[0].view(size)).T, cmap='gray')
    ax03 = fig.add_subplot(gs[0, 3]); ax03.imshow(norm(interpolated_bicubic[0].view(size)).T, cmap='gray')

    # Row 1 (second row of HF images)
    ax10 = fig.add_subplot(gs[1, 0]); ax10.imshow(norm(hf_ground_truth_list[1].T), cmap='gray', vmin=0, vmax=1)
    ax11 = fig.add_subplot(gs[1, 1]); ax11.imshow(norm(get_full_img(model_output_seg_list[1], model_output_img_list[1], config)).cpu().view(size).detach().numpy().T, cmap='gray')
    ax12 = fig.add_subplot(gs[1, 2]); ax12.imshow(norm(interpolated_bilinear[1].view(size)).T, cmap='gray')
    ax13 = fig.add_subplot(gs[1, 3]); ax13.imshow(norm(interpolated_bicubic[1].view(size)).T, cmap='gray')

    # Row 3 (LF images)
    ax30 = fig.add_subplot(gs[3, 0]); ax30.imshow(norm(lf_gt[0].view(size_lf).detach().cpu()).T, cmap='gray')
    ax31 = fig.add_subplot(gs[3, 1]); ax31.imshow(norm(lf_output[0].view(size_lf).detach().cpu()).T, cmap='gray')
    ax32 = fig.add_subplot(gs[3, 2]); ax32.imshow(norm(lf_gt[1].view(size_lf).detach().cpu()).T, cmap='gray')
    ax33 = fig.add_subplot(gs[3, 3]); ax33.imshow(norm(lf_output[1].view(size_lf).detach().cpu()).T, cmap='gray')

    axes = [[ax00, ax01, ax02, ax03],
            [ax10, ax11, ax12, ax13],
            [ax30, ax31, ax32, ax33]]

    # Titles
    font_size = 6
    for ax, title, color in zip([ax00, ax01, ax02, ax03],
                                 ["Ground Truth", "SIREN", "Bilinear", "Bicubic"],
                                 ['blue', 'red', 'blue', 'blue']):
        ax.set_title(title, fontsize=font_size, color=color)

    for ax, title, color in zip([ax30, ax31, ax32, ax33],
                                 ["Ground Truth", "SIREN", "Ground Truth", "SIREN"],
                                 ['blue', 'red', 'blue', 'red']):
        ax.set_title(title, fontsize=font_size, color=color)

    # Turn off axes
    for row in axes:
        for ax in row:
            ax.axis('off')
            ax.invert_yaxis()

    sub_color = 'darkviolet'
    # Row annotations
    ax00.annotate('Subject = 1', (-0.15, 0.5), xycoords='axes fraction', rotation=90, va='center', fontsize=font_size, color=sub_color)
    ax10.annotate('Subject = 2', (-0.15, 0.5), xycoords='axes fraction', rotation=90, va='center', fontsize=font_size, color=sub_color)

    # Column-side annotations
    fig.text(-0.00001, 0.675, "High-Field", ha='left', va='center', rotation='vertical', fontsize=font_size, color='blue')
    fig.text(-0.00001, 0.175, "Ultra Low-Field", ha='left', va='center', rotation='vertical', fontsize=font_size, color='blue')

    # Bottom annotations
    
    fig.text(0.21, 0.35, "Subject = 1", ha='left', va='center', rotation='horizontal', fontsize=7, color=sub_color)
    fig.text(0.725, 0.35, "Subject = 2", ha='left', va='center', rotation='horizontal', fontsize=7, color=sub_color)

    
    # #Image Metrics annotation
    score_siren = [df["PSNR1"][0], df["SSIM1"][0], df["PSNR2"][0], df["SSIM2"][0]]
    score_bicubic = [df["PSNR1"][1], df["SSIM1"][1], df["PSNR2"][1], df["SSIM2"][1]]
    score_bilinear = [df["PSNR1"][2], df["SSIM1"][2], df["PSNR2"][2], df["SSIM2"][2]]


    ax01.annotate(f"PSNR: {score_siren[0]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax01.annotate(f"SSIM: {score_siren[1]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    # ax02.annotate(f"PSNR: {scores_subject1[1]['psnr']:.2f}           SSIM: {scores_subject1[0]['ssim']:.3f}", (0.05, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax02.annotate(f"PSNR: {score_bilinear[0]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax02.annotate(f"SSIM: {score_bilinear[1]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')

    ax03.annotate(f"PSNR: {score_bicubic[0]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax03.annotate(f"SSIM: {score_bicubic[1]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')


    ax11.annotate(f"PSNR: {score_siren[2]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax11.annotate(f"SSIM: {score_siren[3]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    # ax02.annotate(f"PSNR: {scores_subject1[1]['psnr']:.2f}           SSIM: {scores_subject1[0]['ssim']:.3f}", (0.05, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax12.annotate(f"PSNR: {score_bilinear[2]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax12.annotate(f"SSIM: {score_bilinear[3]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')

    ax13.annotate(f"PSNR: {score_bicubic[2]:.2f}", (0.625, 0.05), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')
    ax13.annotate(f"SSIM: {score_bicubic[3]:.3f}", (0.625, 0.95), xycoords='axes fraction', rotation=0, va='center', fontsize=font_size-1, color='lime')

    




    # Get positions of ax31 and ax32
    pos1 = ax31.get_position()
    pos2 = ax32.get_position()

    # Compute x position for vertical line (midpoint between two axes)
    x_line = (pos1.x1 + pos2.x0) / 2

    # Define y range for the line (row 3 vertical range)
    y_min = pos1.y0
    y_max = pos1.y1

    # Add the vertical line to the figure
    line = Line2D([x_line, x_line], [y_min, y_max], transform=fig.transFigure,
                color=sub_color, linewidth=1, linestyle='--')
    fig.add_artist(line)

    fig.tight_layout()
    return fig



import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_seg_results_compare_2_paper(config, model_output_seg_list, hf_gt_seg_list, lf_gt_seg_list, prior_seg_dice):
    
    size = config["size"]
    size_lf = config["size_lf"]

    # model_output_seg1 = model_output_seg_list[0]
    fig, axes = plt.subplots(6,3, figsize = (4,6), layout='constrained')
    im = im00 = axes[0,0].imshow(hf_gt_seg_list[0][0,1,:,:].cpu().T, cmap='gray', vmin=0,vmax=1)
    im = im01 = axes[0,1].imshow(hf_gt_seg_list[0][0,2,:,:].cpu().T, cmap='gray', vmin=0,vmax=1)
    im = im02 = axes[0,2].imshow(hf_gt_seg_list[0][0,3,:,:].cpu().T, cmap='gray', vmin=0,vmax=1)
    # im = im03 = axes[0,3].imshow(hf_gt_seg_list[0][0,0,:,:].cpu(), cmap='gray', vmin=0,vmax=1)

    im = im1 = axes[1,0].imshow(model_output_seg_list[0][:,:,0].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    im = im2 = axes[1,1].imshow(model_output_seg_list[0][:,:,1].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    im = im3 = axes[1,2].imshow(model_output_seg_list[0][:,:,2].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    # im = im4 = axes[1,3].imshow(model_output_seg_list[0][:,:,3].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)

    # im = im5 = axes[2,0].imshow(model_output_seg_list[1][:,:,0].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im6 = axes[2,1].imshow(model_output_seg_list[1][:,:,1].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im7 = axes[2,2].imshow(model_output_seg_list[1][:,:,2].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im8 = axes[2,3].imshow(model_output_seg_list[1][:,:,3].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    
    im = im13 = axes[2,0].imshow(lf_gt_seg_list[0][0,1,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im14 = axes[2,1].imshow(lf_gt_seg_list[0][0,2,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im15 = axes[2,2].imshow(lf_gt_seg_list[0][0,3,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    # im = im16 = axes[2,3].imshow(lf_gt_seg_list[0][0,0,:,:].cpu().detach().numpy(), cmap='gray', vmin=0,vmax=1)

    #subject = 2
    im = im100 = axes[3,0].imshow(hf_gt_seg_list[1][0,1,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im101 = axes[3,1].imshow(hf_gt_seg_list[1][0,2,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im102 = axes[3,2].imshow(hf_gt_seg_list[1][0,3,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    # im = im103 = axes[3,3].imshow(hf_gt_seg_list[1][0,0,:,:].cpu().detach().numpy(), cmap='gray', vmin=0,vmax=1)

    

    im = im111 = axes[4,0].imshow(model_output_seg_list[1][:,:,0].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    im = im112 = axes[4,1].imshow(model_output_seg_list[1][:,:,1].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    im = im113 = axes[4,2].imshow(model_output_seg_list[1][:,:,2].cpu().view(size).detach().numpy().T, cmap = 'gray',vmin=0,vmax=1)
    # im = im114 = axes[4,3].imshow(model_output_seg_list[1][:,:,3].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)

    # im = im115 = axes[6,0].imshow(model_output_seg_list[3][:,:,0].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im116 = axes[6,1].imshow(model_output_seg_list[3][:,:,1].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im117 = axes[6,2].imshow(model_output_seg_list[3][:,:,2].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    # im = im118 = axes[6,3].imshow(model_output_seg_list[3][:,:,3].cpu().view(size).detach().numpy(), cmap = 'gray',vmin=0,vmax=1)
    
    im = im119 = axes[5,0].imshow(lf_gt_seg_list[1][0,1,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im120 = axes[5,1].imshow(lf_gt_seg_list[1][0,2,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    im = im121 = axes[5,2].imshow(lf_gt_seg_list[1][0,3,:,:].cpu().detach().numpy().T, cmap='gray', vmin=0,vmax=1)
    # im = im122 = axes[5,3].imshow(lf_gt_seg_list[1][0,0,:,:].cpu().detach().numpy(), cmap='gray', vmin=0,vmax=1)


    # imx = im9 = axes[8,0].imshow(prior_seg_dice[0,1,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    # imx = im10 = axes[8,1].imshow(prior_seg_dice[0,2,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    # imx = im11 = axes[8,2].imshow(prior_seg_dice[0,3,:,:].cpu().view(size).detach().numpy(), cmap='gray')
    # imx = im12 = axes[8,3].imshow(prior_seg_dice[0,0,:,:].cpu().view(size).detach().numpy(), cmap='gray')



    for i in range(0,6):
        for j in range(0,3):
            axes[i,j].set_axis_off()
            axes[i,j].invert_yaxis()
            
            
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=1 / 72, hspace=0.01, wspace=0.01)    

    # cbar = fig.colorbar(im, ax=axes[:8,:] ,shrink = 0.9, aspect=55*0.9, pad =0.01)
    # cbar2 = fig.colorbar(imx, ax=axes[8:,:] ,shrink = 0.9, aspect=55*0.9, pad =0.01)
    
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs, fontsize=6)
    # ticklabs2 = cbar2.ax.get_yticklabels()
    # cbar2.ax.set_yticklabels(ticklabs2, fontsize=6)
    # fig.subplots_adjust(right=1.2)

    sub_color = 'darkviolet'
    axes[0,0].set_title("White Matter", color='blue', fontsize = 8)
    axes[0,1].set_title("Gray Matter", color='blue', fontsize = 8)
    axes[0,2].set_title("CSF", color='blue', fontsize = 8)
    # axes[0,3].set_title("Background", color='blue', fontsize = 8)
    # axes[0,:].set_title("Subject = 1 ", color='brown', fontsize = 8)
    
    
    # axes[0,0].set_title("GT", color='blue', fontsize = 9, rotation='vertical', x = -0.05, y =0.5)
    axes[0,0].annotate('Ground Truth', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='blue')
    axes[1,0].annotate('SIREN', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='red')
    # axes[2,0].annotate('SIREN+FFE', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='red')
    axes[2,0].annotate('ULF-Ground Truth', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='blue')
    


    axes[3,0].annotate('Ground Truth', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='blue')
    axes[4,0].annotate('SIREN', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='red')
    # axes[6,0].annotate('SIREN+FFE', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='red')
    axes[5,0].annotate('ULF-Ground Truth', (-0.1, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='blue')
    # axes[6,0].annotate('Prior', (-0.15, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontsize = 6, color='blue')
    # fig.tight_layout()
    
    fig.text(-0.00001, 0.8, "Subject = 1", ha='left', va='center', rotation='vertical', fontsize=6, color = sub_color)
    fig.text(-0.00001, 0.3, "Subject =  2", ha='left', va='center', rotation='vertical', fontsize=6, color = sub_color)
    # fig.text(0.01, 0.05, "Title Group 3", ha='left', va='center', rotation='vertical', fontsize=8, color = 'brown')
    
    y = 0.49  # approximately halfway (adjust as needed)

    # Create horizontal line from left to right (x=0 to x=1) at y
    line = Line2D([-0.0235, 1], [y, y], transform=fig.transFigure, color=sub_color, linewidth=0.75, linestyle='--')
    fig.add_artist(line)
    return fig
