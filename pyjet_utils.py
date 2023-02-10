import numpy as np
from pyjet import DTYPE_PTEPM, cluster
from matplotlib import colors
import matplotlib.pyplot as plt



class ConstructJet(object):
    def __init__(self, image, R=0.5, p=-1, ptmin=20):
        self.image = image.reshape(1, 72, 72)
        #self.eta = np.linspace(-4.337, 4.337, 72)
        #self.phi = np.linspace(-np.pi, np.pi, 73)[:-1]
        self.eta = np.load("eta_bins.npy")[:-1]
        self.phi = np.load("phi_bins.npy")[:-1]
        Eta = np.ones_like(self.image)*self.eta.reshape(-1, 1)
        Phi = np.ones_like(self.image)*self.phi.reshape(1, -1)
        Mass = np.zeros_like(self.image)
        event_tensor = np.concatenate([self.image, Eta, Phi, Mass], axis=0)

        def get_event(event_tensor):
            desired_pts = np.tile(self.image > 0, (4, 1, 1))
            masked_event_tensor = event_tensor[desired_pts].reshape(4, -1).T
            event = np.asarray([tuple(pt_eta_phi_m) for pt_eta_phi_m in masked_event_tensor.tolist()], dtype=DTYPE_PTEPM)
            return event
        event = get_event(event_tensor)

        def get_jets(event, R=R, p=p, ptmin=ptmin):
            sequence = cluster(event, R=R, p=p)
            jets = sequence.inclusive_jets(ptmin=ptmin)
            return jets
        jets = get_jets(event)
        self.event = event
        self.jets = jets
        return


    def get_jets_pts(self, jets_no=[0, 1]):
        jets = self.jets
        jets_pts = []
        for jet_no in jets_no:
            if jet_no >= len(jets):
                returned_pt = 0
                message = "Jet no {} not found! Returned pt: {}".format(jet_no, returned_pt)
                print(message)
                jets_pts.append(returned_pt)
                continue
            jet = jets[jet_no]
            jets_pts.append(jet.pt)
        return np.array(jets_pts)


    def get_eta_difference(self):
        if len(self.jets) < 2:
            returned_eta_diff = 10
            message = "Found jets={} less than 2! Returned eta_diff: {}".format(
            len(self.jets), returned_eta_diff)
            print(message)
            return np.array([returned_eta_diff])
        eta_difference = self.jets[0].eta - self.jets[1].eta
        return np.array([eta_difference])


    def get_phi_difference(self):
        if len(self.jets) < 2:
            returned_phi_diff = 10
            message = "Found jets={} less than 2! Returned phi_diff: {}".format(
            len(self.jets), returned_phi_diff)
            print(message)
            return np.array([returned_phi_diff])
        phi_difference = self.jets[0].phi - self.jets[1].phi
        return np.array([phi_difference])


    def get_eta_and_phi_values(self, jets_no=[0, 1]):
        etas = []
        phis = []
        for jet_no in jets_no:
            if jet_no >= len(self.jets):
                returned_value = 10
                print("Jet no {} not found returned value: {}".format(
                jet_no, returned_value))
                etas.append(returned_value)
                phis.append(returned_value)
                continue
            etas.append(self.jets[jet_no].eta)
            phis.append(self.jets[jet_no].phi)
        return np.array(etas), np.array(phis)


    def get_jet_information(self, jet_no=0):
        jets = self.jets
        jet = jets[jet_no]
        constituents = jet.constituents()

        def get_phi_location(constituent):
            phi_extended = np.concatenate([self.phi, np.array([np.pi])], axis=0)
            phi_location = np.abs(phi_extended - constituent.phi).argmin() % 72
            return phi_location

        def get_eta_location(constituent):
            eta_location = np.abs(self.eta - constituent.eta).argmin()
            return eta_location
        const_locations = []
        const_pts = []
        for constituent in constituents:
            col = get_phi_location(constituent)
            row = get_eta_location(constituent)
            const_locations.append((row, col))
            const_pts.append(constituent.pt)
        jet_pt = jet.pt
        jet_location = get_eta_location(jet), get_phi_location(jet)
        return {
            "constituent_pts": const_pts,
            "constituent_locations":const_locations,
            "jet_pt": jet_pt,
            "jet_location": jet_location
            }


    def locate_jet_on_image(self, jet_no=0):
        jet_image = np.zeros_like(self.image.reshape(72, 72))
        info_dict = self.get_jet_information(jet_no=jet_no)
        pts, locations = info_dict["constituent_pts"], info_dict["constituent_locations"]
        for i in range(len(locations)):
            row, col = locations[i]
            pt = pts[i]
            jet_image[row, col] = pt
        return jet_image


    def locate_specific_jets_on_image(self, jets_no=[0, 1]):
        im_shape = 72, 72
        jets_image = np.zeros_like(self.image.reshape(im_shape))
        for jet_no in jets_no:
            if jet_no >= len(self.jets):
                print("jet no {} not found!".format(jet_no))
                continue
            jets_image += self.locate_jet_on_image(jet_no)
        return jets_image


    def get_jet_images_as_channels(self, jets_no):
        im_shape = (72, 72)
        jet_images = []
        for jet_no in jets_no:
            if jet_no >= len(self.jets):
                jet_images.append(np.zeros(im_shape))
                continue
            jet_images.append(self.locate_jet_on_image(jet_no))
        return np.array(jet_images)


    def plot_jets(self, jets_no=[0, 1], fig=None, ax=None, vmax=None, eps=1e-5):
        if vmax==None:
            vmax = self.image.max()
        colors_list = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b', '#e377c2',
            '#9467bd', '#2ca02c', '#7f7f7f', '#bcbd22', '#17becf']
        jets_image = self.locate_specific_jets_on_image(jets_no)
        im_s = ax.imshow(jets_image + eps, norm=colors.LogNorm(vmax=vmax))
        fig.colorbar(im_s, ax=ax, shrink=0.9, orientation="horizontal", pad=0.01)
        for jet_no in jets_no:
            if jet_no >= len(self.jets):
                print("jet no {} not found!".format(jet_no))
                continue
            info_dict = self.get_jet_information(jet_no)
            scatter_label = "jet: {}, pt={:.0f}".format(jet_no, info_dict["jet_pt"])
            row, col = info_dict["jet_location"]
            ax.scatter(col, row, label=scatter_label, color=colors_list[jet_no])
        ax.legend()
        ax.axis("off")
        return


    def plot_original_image(self, fig=None, ax=None, vmax=None, eps=1e-5):
        if vmax==None:
            vmax = self.image.max()
        im_s = ax.imshow(self.image.reshape(72, 72) + eps, norm=colors.LogNorm(vmax=vmax))
        fig.colorbar(im_s, ax=ax, shrink=0.9, orientation="horizontal", pad=0.01)
        ax.axis("off")
        return


    def show_original_and_jets_images(self, jets_no=[0, 1]):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
        ax[0].set(title="image")
        self.plot_original_image(fig, ax[0])
        ax[1].set(title="jets")
        self.plot_jets(jets_no, fig, ax[1])
        fig.show()
        return
