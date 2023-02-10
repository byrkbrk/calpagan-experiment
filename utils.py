import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from time import time
import pickle
from tqdm.auto import tqdm
from pyjet_utils import ConstructJet



def create_directories():
    cur_dir = os.getcwd()
    print(f"Current working dir: {cur_dir}")
    image_dir = "images"
    gen_params_dir = "gen_parameters"
    disc_params_dir = "disc_parameters"
    os.mkdir(image_dir)
    os.mkdir(gen_params_dir)
    os.mkdir(disc_params_dir)
    print(f"Created dirs: {image_dir}, {gen_params_dir}, {disc_params_dir}")


def save_gen_params(generator, cur_step):
    gen_params_dir = "gen_parameters"
    file_name = gen_params_dir + "/gen_state_dict_cur_step=" + str(cur_step) + ".pt"
    torch.save(generator.state_dict(), file_name)


def save_disc_params(discriminator, cur_step):
    disc_params_dir = "disc_parameters"
    file_name = disc_params_dir + "/disc_state_dict_cur_step=" + str(cur_step) + ".pt"
    torch.save(discriminator.state_dict(), file_name)


def convert_notebook():
        import re
        list_dir = os.listdir(".")
        print(list_dir)
        for file in list_dir:
            if re.search(r"pix2pix.*\.ipynb$", file):
                notebook_name = file
                print("found notebook name:", notebook_name)
                break
        #notebook_name = "pix2pix_dijet_72x72"
        cmd_html = "jupyter nbconvert --to html " + notebook_name
        result = subprocess.run(cmd_html.split())
        if result.returncode == 0:
            print("Converting successful")
            print(notebook_name)
        else:
            print("Converting failed!")


def show_tensor_image(condition, real, fake, eps=1e-5):
    im_shape = (72, 72)
    condition = condition.reshape(im_shape).cpu()
    real = real.reshape(im_shape).cpu()
    fake = fake.detach().reshape(im_shape).cpu()
    relu = nn.ReLU()
    fake = relu(fake)

    #eps = 1e-4
    vmax = torch.max(torch.cat([condition, real, fake])).item()

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(condition+eps, cmap="viridis", norm=colors.LogNorm(vmax=vmax))
    plt.colorbar(shrink=0.9, orientation="horizontal", pad=0.01)
    plt.title("Condition")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(real+eps, cmap="viridis", norm=colors.LogNorm(vmax=vmax))
    plt.colorbar(shrink=0.9, orientation="horizontal", pad=0.01)
    plt.title("Real")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fake+eps, cmap="viridis", norm=colors.LogNorm(vmax=vmax))
    plt.colorbar(shrink=0.9, orientation="horizontal", pad=0.01)
    plt.title("Fake")
    plt.axis("off")

    plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, hspace=0.1, wspace=0.1)
    plt.show()


def show_tensor_images(conditions, reals, fakes, eps=1e-5):
    showing_count = 8
    for im_no in range(len(reals)):
        condition, real, fake = conditions[im_no], reals[im_no], fakes[im_no]
        show_tensor_image(condition, real, fake, eps)
        if im_no == 7:
            break


def show_single_image_and_jets(condition, real, fake, jets_no=[0, 1], eps=1e-5, show=True, save=False, file_name=None):
    im_shape = (72, 72)
    condition = condition.reshape(im_shape).cpu().numpy()
    real = real.reshape(im_shape).cpu().numpy()
    fake = fake.detach().reshape(im_shape).cpu()
    relu = nn.ReLU()
    fake = relu(fake).numpy()

    c_construct_jet = ConstructJet(condition)
    r_construct_jet = ConstructJet(real)
    f_construct_jet = ConstructJet(fake)

    vmax = np.concatenate([condition, real, fake], axis=0).max()

    fig, (ax_orig, ax_jet) = plt.subplots(2, 3, figsize=(15, 12))
    ax_orig[0].set(title="Condition")
    c_construct_jet.plot_original_image(fig, ax_orig[0], vmax, eps)
    ax_jet[0].set(title="Condition jets")
    c_construct_jet.plot_jets(jets_no, fig, ax_jet[0], vmax, eps)

    ax_orig[1].set(title="Real")
    r_construct_jet.plot_original_image(fig, ax_orig[1], vmax, eps)
    ax_jet[1].set(title="Real jets")
    r_construct_jet.plot_jets(jets_no, fig, ax_jet[1], vmax, eps)

    ax_orig[2].set(title="Fake")
    f_construct_jet.plot_original_image(fig, ax_orig[2], vmax, eps)
    ax_jet[2].set(title="Fake jets")
    f_construct_jet.plot_jets(jets_no, fig, ax_jet[2], vmax, eps)

    if save:
        fig.savefig(file_name)
    if show:
        plt.show()
    plt.close(fig)
    return


def show_multiple_images_and_jets(conditions, reals, fakes, cur_step, jets_no=[0, 1], show_n_im=8, eps=1e-5, show=True, save=False):
    for im_no in range(len(reals)):
        if im_no == show_n_im:
            break
        condition = conditions[im_no]
        real = reals[im_no]
        fake = fakes[im_no]
        print("im_no:", im_no)
        file_name = f"images/cur_step={cur_step}_im_no={im_no}.png"
        show_single_image_and_jets(condition, real, fake, jets_no, eps, show, save, file_name)
        print()
    return


def show_loss_curve(loss_dict, cur_step, mean_disc_loss, mean_adv_loss, mean_recon_loss=None, mean_gen_total_loss=None, show=True, save=False):
    loss_dict["steps"].append(cur_step)
    loss_dict["disc_losses"].append(mean_disc_loss)
    loss_dict["adv_losses"].append(mean_adv_loss)
    loss_dict["recon_losses"].append(mean_recon_loss)
    loss_dict["gen_losses"].append(mean_gen_total_loss)
    steps, disc_losses, adv_losses = loss_dict["steps"], loss_dict["disc_losses"], loss_dict["adv_losses"]
    recon_losses, gen_losses = loss_dict["recon_losses"], loss_dict["gen_losses"]
    plt.plot(
        steps, disc_losses, label="Discriminator"
    )
    plt.plot(
        steps, adv_losses, label="Gen adversarial"
    )
    if mean_recon_loss != None:
        plt.plot(
            steps, recon_losses, label="Reconstruction"
        )
    if mean_gen_total_loss != None:
        plt.plot(
            steps, gen_losses, label="Total generator"
        )
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    if show:
        plt.show()
    if save:
        image_name = "loss_curve_cur_step=" + str(steps[-1])
        plt.savefig(image_name)
    return loss_dict


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    from scipy.linalg import sqrtm
    def matrix_sqrt(x):
        y = x.cpu().detach().numpy()
        y = sqrtm(y) #scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device=x.device)
    norm_square = torch.norm(mu_x - mu_y)**2
    trace = torch.trace(sigma_x + sigma_y - 2*matrix_sqrt(sigma_x @ sigma_y))
    return norm_square + trace


def get_fid_score(features_x, features_y):
    mu_x = features_x.mean(axis=0)
    mu_y = features_y.mean(axis=0)

    def get_covariance(features):
        return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))
    sigma_x = get_covariance(features_x)
    sigma_y = get_covariance(features_y)
    return frechet_distance(mu_x, mu_y, sigma_x, sigma_y)


def sample_images(dataset, gen, device):
    """ 
    Inputs:
        dataset: zip(geant_normalized, delphes_normalized, max_each_delphes)
        gen: generator
        device: gpu or cpu
    Outputs:
        geant-features, delphes-features, fake-features (each dims=(n, -1))
    """
    batch_size = 64
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
    )
    geant_ims = torch.Tensor([])
    delphes_ims = torch.Tensor([])
    fake_ims = torch.Tensor([])
    for batch in dataloader:
        geant_im, delphes_im, max_each_delphes = batch
        fake_im = gen(torch.cat([delphes_im.to(device), torch.rand_like(delphes_im, device=device)], axis=1)).detach().cpu()
        
        # go back to original scale
        geant_im = geant_im*max_each_delphes
        delphes_im = delphes_im*max_each_delphes
        fake_im = fake_im*max_each_delphes
        
        # concatanete
        geant_ims = torch.cat([geant_ims, geant_im])
        delphes_ims = torch.cat([delphes_ims, delphes_im])
        fake_ims = torch.cat([fake_ims, fake_im])
    
    # reshape features 
    n_images = geant_ims.shape[0]
    geant_ims = geant_ims.reshape(n_images, -1)
    delphes_ims = delphes_ims.reshape(n_images, -1)
    fake_ims = fake_ims.reshape(n_images, -1)
    
    return geant_ims, delphes_ims, fake_ims


def show_fid_curve(dataset, generator, cur_step, fid_dict, device):
    tic = time()
    delta_t = lambda tic: (time()-tic)/60
    
    # update steps
    fid_dict["steps"].append(cur_step)
    steps = fid_dict["steps"]
    sample_geant, sample_delphes, sample_fake = sample_images(dataset, generator, device)
    print("sample images delta_t (in min):", delta_t(tic))
    if len(steps) == 1:
        
        # update Geant-Delphes
        fid_dict["geant_delphes"] += [get_fid_score(sample_geant, sample_delphes).item()]
        print("at G-D dt:", delta_t(tic))
    else:
        
        # update Geant-Delphes
        fid_dict["geant_delphes"] += [fid_dict["geant_delphes"][0]]
    
    # update Geant-Fake
    fid_dict["geant_fake"] += [get_fid_score(sample_geant, sample_fake).item()]
    print("at G-F dt:", delta_t(tic))
    
    # update Delphes-Fake
    fid_dict["delphes_fake"] += [get_fid_score(sample_delphes, sample_fake).item()]
    print("at D-F dt:", delta_t(tic))
    if len(steps) > 5:
        fid_geant_delphes, fid_geant_fake, fid_delphes_fake = fid_dict["geant_delphes"], fid_dict["geant_fake"], fid_dict["delphes_fake"]
        plt.plot(steps[4:], fid_geant_delphes[4:], label="geant-delphes")
        plt.plot(steps[4:], fid_geant_fake[4:], label="geant-fake")
        plt.plot(steps[4:], fid_delphes_fake[4:], label="delphes-fake")
        plt.legend()
        plt.xlabel("steps")
        plt.ylabel("fid score")
        plt.show()
        print("G-D: {}\nG-F: {}\nD-F: {}".format(fid_geant_delphes[-1], fid_geant_fake[-1], fid_delphes_fake[-1]))
    return fid_dict


def get_shift_info(tensor_image):
    """Get one tensor image as input and returns shift information to roll the matrix
    to get maximum valued pixel at the center pixel.
    Input:
    tensor_image: torch.tensor having shape (1, 1, h, w)
    Output:
    shift_info: a tuple (., .); first coordinate is shift info of height and
    second is shift info of width.
    """
    _, h, w = tensor_image.shape
    center_h = (h-1)//2
    center_w = (w-1)//2
    max_pixel_position = tensor_image.argmax()
    max_pixel_h, max_pixel_w = max_pixel_position // h, max_pixel_position % w
    return center_h - max_pixel_h, center_w - max_pixel_w


def shift_image(tensor_image, shift_info):
    """Gets one tensor image and corresponding shift info and returns rolled image
    Inputs:
    tensor_image: torch.tensor having a shape (1, 1, h, w)
    shift_info: a tuple having length 2 (i.e. len(shift_info)=2) that contains info of
    how to shift the height and width of the image.
    Output:
    rolled_image: torch.tensor having the same shape with tensor_image.
    """
    return tensor_image.roll(shifts=shift_info, dims=(1, 2))


def get_shifts(tensor_images):
    """Get tensor images as input and returns shift information for each image in a list
    input:
    tensor_images: torch.tensor having shape (-1, 1, h, w)
    output:
    shifts: list containig roll information for each image in tensor images
    """
    shifts = []
    for tensor_image in tensor_images:
        shifts.append(get_shift_info(tensor_image))
    return shifts


def shift_images(tensor_images, shifts):
    """Get tensor images and correspondin shift information as input and returns rolled images
    inputs:
    tensor_images: torch.tensor having shape (-1, 1, h, w)
    shifts: list containing shift info for each image
    output:
    shifted_images: torch.tensor having shape (-1, 1, h, w)
    """
    shifted_images = torch.zeros_like(tensor_images)
    for i in range(len(tensor_images)):
        tensor_image, shift = tensor_images[i], shifts[i]
        shifted_images[i] = shift_image(tensor_image, shift)
    return shifted_images


def crop(tensor_images, new_shape):
    h = new_shape[2]
    w = new_shape[3]
    h_start = int((tensor_images.shape[2] - h) / 2)
    w_start = int((tensor_images.shape[3] - w) / 2)
    cropped_images = tensor_images[:, :, h_start:h_start+h, w_start:w_start+w]
    return cropped_images


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def load_model(generator, gen_param_path):
    state_dict_name = gen_param_path
    gen_state_dict = torch.load(state_dict_name)
    generator.load_state_dict(gen_state_dict)
    return generator


def normalize_by_example(A, B):
    n_examples = len(A)
    max_A = A.reshape(n_examples, -1).max(axis=1).values
    max_B = B.reshape(n_examples, -1).max(axis=1).values

    t_values = max_A > max_B
    normalization_constants = max_A*t_values + max_B*(~t_values)
    normalization_constants = normalization_constants.reshape(n_examples, 1, 1, 1)
    normalized_A = A/normalization_constants
    normalized_B = B/normalization_constants
    return normalized_A, normalized_B
