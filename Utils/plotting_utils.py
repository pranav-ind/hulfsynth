from matplotlib import pyplot as plt
import torch
import numpy as np

size = (87*2, 96*2)
size_lf = (87,96)

def plot_2_images(img1, img2, title1="img1", title2="img2", ):
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    im1 = axes[0].imshow(img1, cmap = 'gray')
    axes[0].set_title(title1)
    fig.colorbar(im1, shrink = 0.6)
    im2 = axes[1].imshow(img2, cmap = 'gray')
    axes[1].set_title(title2)
    fig.colorbar(im2,  shrink = 0.6)
    # plt.show(block=False)
    return fig


def plot_4_images(img1, img2, img3, img4, title1="img1", title2="img2", title3="img3", title4="img4"):

    fig, axes = plt.subplots(1,4, figsize=(20,6))
    im1 = axes[0].imshow(img1, cmap = 'gray')
    im2 = axes[1].imshow(img2, cmap = 'gray')
    im3 = axes[2].imshow(img3, cmap = 'gray')
    im4 = axes[3].imshow(img4, cmap = 'gray')
    
    fig.colorbar(im1, shrink = 0.6)
    fig.colorbar(im2, shrink = 0.6)
    fig.colorbar(im3, shrink = 0.6)
    fig.colorbar(im4, shrink = 0.6)

    axes[0].set_title(title1)
    axes[1].set_title(title2)
    axes[2].set_title(title3)
    axes[3].set_title(title4)

    # plt.show(block=False)
    return fig

def plot_5_images(img1, img2, img3, img4, img5, title1="img1", title2="img2", title3="img3", title4="img4", title5="img5"):

    fig, axes = plt.subplots(1,5, figsize=(20,6))
    im1 = axes[0].imshow(img1, cmap = 'gray')
    im2 = axes[1].imshow(img2, cmap = 'gray')
    im3 = axes[2].imshow(img3, cmap = 'gray')
    im4 = axes[3].imshow(img4, cmap = 'gray')
    im5 = axes[4].imshow(img5, cmap = 'gray')
    # im6 = axes[4].imshow(img5, cmap='Reds' ,alpha= 0.9*(condition))
    
    fig.colorbar(im1, shrink = 0.4)
    fig.colorbar(im2, shrink = 0.4)
    fig.colorbar(im3, shrink = 0.4)
    fig.colorbar(im4, shrink = 0.4)
    fig.colorbar(im5, shrink = 0.4)
    # fig.colorbar(im6, shrink = 0.4)

    axes[0].set_title(title1)
    axes[1].set_title(title2)
    axes[2].set_title(title3)
    axes[3].set_title(title4)
    axes[4].set_title(title5)

    # plt.show(block=False)
    return fig

"""
def loss_plot(loss_dict):
    plt.plot(range(len(loss_dict["L1"])), loss_dict["L1"],label = "L1" )
    plt.plot(range(len(loss_dict["L2"])), loss_dict["L2"],label = "L2" )
    plt.plot(range(len(loss_dict["Total_Loss"])), loss_dict["Total_Loss"],label = "Total Loss" )
    plt.legend()
    plt.show()

def loss_plot(loss_dict):
    for key in loss_dict.keys():
        plt.plot(range(len(loss_dict[key])), loss_dict[key],label = key)
    plt.legend()
    plt.show()
"""


def loss_plot(loss_dict):
    num_plots = len(loss_dict.keys())
    fig, axes = plt.subplots(1,num_plots, figsize=(20,5))
    # colors = ['r', 'b', 'g', 'k', 'm', '']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, key in enumerate(loss_dict.keys()):
        axes[i].plot(range(len(loss_dict[key])), loss_dict[key],label = key, color = colors[i])
        axes[i].set_xlabel("Epochs",color='blue')
        axes[i].set_ylabel(key.upper(), color='blue')
    fig.legend(loc="upper right")
    plt.tight_layout()
    return fig




def plot_results_after_train(model_output_img, model_output_seg, lf_output, loss_dict):
    
    hf_output = model_output_img.clip(min=0).cpu()
    model_output_seg_new = model_output_seg.to(torch.float32).cpu().detach()
    hf_result = (hf_output[:,:,0] * model_output_seg_new[:,:,0]) + (hf_output[:,:,1]* model_output_seg_new[:,:,1]) + (hf_output[:,:,2] * model_output_seg_new[:,:,2]) + (hf_output[:,:,3] * model_output_seg_new[:,:,3])
    seg_result = model_output_seg_new[:,:,0] + model_output_seg_new[:,:,1] + model_output_seg_new[:,:,2] + model_output_seg_new[:,:,3]
    wm_seg_pred, gm_seg_pred, csf_seg_pred, bg_seg_pred = model_output_seg_new[:,:,0].cpu().view(size).detach().numpy(), model_output_seg_new[:,:,1].cpu().view(size).detach().numpy(), model_output_seg_new[:,:,2].cpu().view(size).detach().numpy(), model_output_seg[:,:,3].cpu().view(size).detach().numpy()

    plot_2_images(lf_output.cpu().view(size_lf).detach().numpy(), hf_result.cpu().view(size).detach().numpy(), "Predicted Low-Field", "Predicted High-Field")
    plot_4_images(wm_seg_pred, gm_seg_pred, csf_seg_pred, bg_seg_pred , "WM-Prediction", "GM-Prediction", "CSF-Prediction", "BG-Prediction")
    loss_plot(loss_dict)



def plot_intermediate_results(model_output_img, lf_output, coords):
    hf_int_result = model_output_img[:,:,0] + model_output_img[:,:,1] + model_output_img[:,:,2] + model_output_img[:,:,3] #Intermediate High-Field Result
    img_grad = gradient(hf_int_result, coords)
    img_laplacian = laplace(hf_int_result, coords)
    clear_output(wait=True)
    fig, axes = plt.subplots(1,4, figsize=(20,6))
    axes[0].imshow(lf_output.cpu().view(size_lf).detach().numpy(), cmap = 'gray')
    axes[1].imshow(hf_int_result.cpu().view(size).detach().numpy(), cmap = 'gray')
    axes[2].imshow(img_grad.norm(dim=-1).cpu().view(size).detach().numpy(), cmap = 'gray')
    axes[3].imshow(img_laplacian.cpu().view(size).detach().numpy(), cmap = 'gray')
    # plt.show(block=False)
    return fig


def plot_image_metrics(train_metrics):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].plot(range(len(train_metrics["psnr"])), train_metrics["psnr"],label = "PSNR", color='green')
    axes[1].plot(range(len(train_metrics["ssim"])), train_metrics["ssim"],label = "SSIM", color='blue')
    axes[0].legend()
    axes[1].legend()
    axes[0].yaxis.set_ticks(np.arange(min(train_metrics["psnr"]), max(train_metrics["psnr"])))#+0.5, 1.25))
    axes[1].yaxis.set_ticks(np.arange(min(train_metrics["ssim"]), max(train_metrics["ssim"])+0.08))#+0.05, 0.08))
    axes[0].set_xlabel("Epochs",color='blue')
    axes[1].set_xlabel("Epochs",color='blue')
    axes[0].set_ylabel("PSNR",color='blue')
    axes[1].set_ylabel("SSIM",color='blue')
    axes[0].tick_params(axis='both', which='major', labelsize=7)
    axes[1].tick_params(axis='both', which='major', labelsize=7)
    fig.suptitle("Image Metrics over training",color='blue')
    # plt.show(block=False)
    return fig