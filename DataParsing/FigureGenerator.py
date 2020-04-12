import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp
import os
import pickle


plt.rcParams.update({"font.size": 25})

class BuildSpectre:
    def __init__(self, fileName, type):
        self.calibration = None
        self.extradata = None
        self.ROI_limit = []
        self.peaks = []


        if type == "MAESTRO":
            self.data, ROIs, self.extradata,self.calibration = self.__importData_gamma(fileName)
        elif type == "AMDC":
            self.data, ROIs, self.extradata,self.calibration = self.__importData_xray(fileName)
        else:
            assert AssertionError(" Le type est inconnue")
        self.calibrated = False if self.calibration is None else True
        self.datax = list(range(0, len(self.data)))
        self.data_noiseless = self.data.copy()
        for coord in ROIs:
            coords = coord.split(" ")
            self.ROI_limit.append((int(coords[0]), int(coords[1])))
        self.noise_array = np.zeros(len(self.data))

    @staticmethod
    def __importData_gamma(fileName: str) -> tuple:
        with open("Data/{}".format(fileName), "r") as datafile:
            rawstr = datafile.read().split("\n")
            ROI_index = rawstr.index("$ROI:")
            temps_ind = rawstr.index("$MEAS_TIM:") +1
            data_list = rawstr[12:ROI_index]
            temp = rawstr[temps_ind].split(" ")
            temps = temp[-1]
            try:
                if int(rawstr[ROI_index+1]) == 0:
                    raise Exception
                ROI_bloc = rawstr[ROI_index + 2:rawstr.index("$PRESETS:")]
            except:
                ROI_bloc = []
        return np.array(data_list, dtype="int"), ROI_bloc,{"LIVE_TIME":temps},None

    @staticmethod
    def __importData_xray(fileName: str) -> tuple:
        with open("Data/{}".format(fileName), "r") as datafile:
            rawstr = datafile.read().split("\n")
            data_index = rawstr.index("<<DATA>>")
            config_index = rawstr.index("<<DPP CONFIGURATION>>")
            data_list = rawstr[data_index + 1:config_index - 1]
            CONFIG_list = rawstr[config_index + 1:rawstr.index("<<DPP CONFIGURATION END>>")] + rawstr[rawstr.index("<<DPP STATUS>>")+1:-2]
            info_dict = {}
            try:
                calibration_index = rawstr.index("<<CALIBRATION>>")
                for elem in rawstr[1:calibration_index]:
                    temp = elem.split(" -")
                    info_dict[temp[0]] = temp[1]
                ROI_index = rawstr.index("<<ROI>>")
                ROI_list = rawstr[ROI_index + 1:data_index]
            except:
                ROI_list = []
            try:
                calibration_index = rawstr.index("<<CALIBRATION>>")
                calib_list = rawstr[calibration_index + 2:data_index]
                canal = []
                energie = []
                for elem in calib_list:
                    temp = elem.split(" ")
                    canal.append(float(temp[0]))
                    energie.append(float(temp[1]))
                calibration = Noise(canal, energie)
            except:
                calibration = None
            for elem in CONFIG_list:
                temp = elem.split(": ")
                info_dict[temp[0]] = temp[1]

        return np.array(data_list, dtype="int"), ROI_list, info_dict,calibration

    def annotate_centroid(self,**kwargs):
        element = kwargs.get("name","")
        for index,peak in enumerate(self.peaks):
            center = peak.mucalib
            func1 = lambda x: self.calibration.func_parametre(x) - center
            x = int(sp.root_scalar(func1, bracket = [0,len(self.datax)]).root)
            y = peak.gaussian_function(x) + self.noise_array[x] #self.data[x]
            ymax = plt.ylim()[1]
            delta = ymax - y
            print(fr"{center:.2f} $\pm$ {3*peak.incertitude[0]:.1}")
            if x <= (len(self.datax) / 4) - 1:
                xmax_quart = int(len(self.datax) / 4) - 1
                x_norm = x/xmax_quart
                ajout = (x_norm + 0.2)
                new_x = self.calibration.predict(xmax_quart * ajout) if ajout < 1 else self.calibration.predict(xmax_quart * (ajout - 1))
                plt.annotate(f"{element}  {center:.2f} keV", (center, y), xytext=(new_x, y + delta / 2),arrowprops={"arrowstyle": "->"})

            elif x <= (len(self.datax) / 2) - 1:
                xmax_demi = int(len(self.datax) / 2)-1
                x_norm = x / xmax_demi
                ajout = (x_norm + 0.2)
                new_x = self.calibration.predict((xmax_demi * ajout)) if ajout < 1 else self.calibration.predict((xmax_demi * (ajout - 1)) + ((len(self.datax) / 4)))
                plt.annotate(f"{element}  {center:.2f} keV", (center, y), xytext=(new_x, (delta * np.random.ranf()) + y),arrowprops={"arrowstyle": "->"})

            elif x <= 3 * (len(self.datax) / 4) - 1:
                xmax_3q = int(3 * len(self.datax) / 4) - 1
                x_norm = x / xmax_3q
                ajout = (x_norm + 0.2)
                new_x = self.calibration.predict((xmax_3q * ajout) + ((len(self.datax) / 2))) if ajout < 1 else self.calibration.predict((xmax_3q * (ajout - 1)) + ((len(self.datax) / 2)))
                plt.annotate(f"{element}  {center:.2f} keV", (center, y), xytext=(new_x, y + delta / 2),arrowprops={"arrowstyle": "->"})

            elif x <= len(self.datax)-1:
                xmax_fin = int(len(self.datax)) - 1
                x_norm = x / xmax_fin
                ajout = (x_norm + 0.2)
                new_x = self.calibration.predict((xmax_fin * ajout) + (3 * (len(self.datax) / 4))) if ajout < 1 else self.calibration.predict((xmax_fin * (ajout - 1)) + (3 * (len(self.datax) / 4)))

                plt.annotate(f"{element}  {center:.2f} keV", (center, y), xytext=(new_x, y + delta / 2),arrowprops={"arrowstyle": "->"})


    def show_Spectre(self, title="",**kwargs):
        data = self.data
        type = kwargs.get("type","bar")
        xlabel = kwargs.get("xlabel","Canaux") if not self.calibrated else kwargs.get("xlabel","Énergie [keV]")
        ylabel = kwargs.get("ylabel", "Nombre d'émission détectées")
        label = kwargs.get("label","")
        color = kwargs.get("color",'#1f77b4')
        if type == "bar":
            plt.bar(x=self.datax, height=data, width=1,label = label,zorder = 2) if label != "" else plt.bar(x=self.datax, height=data, width=1,zorder = 2)
        elif type == "plot":
            plt.plot(self.datax,data,label= label,alpha = 0.8,zorder = 2) if label != "" else plt.plot(self.datax,data,alpha = 0.8,zorder = 2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}")
        plt.margins(x=0, y=0)

    def save_fig(self, figureName: str) -> None:
        plt.savefig(figureName)

    def add_ROIS_to_fig(self,**kwargs):
        ROI_mask = np.full(len(self.data),np.nan)
        label = kwargs.get("label","")
        type = kwargs.get("type","bar")
        for coords in self.ROI_limit:
            ROI_mask[coords[0]:coords[1] + 1] = self.data[coords[0]:coords[1] + 1]
            # if not self.calibrated:
            #     ROI_mask[coords[0]:coords[1] + 1] = self.data[coords[0]:coords[1] + 1]
            # else:
            #     func1 = lambda x : self.calibration.func_parametre(x) - coords[0]
            #     fun2 = lambda x : self.calibration.func_parametre(x) - coords[1]
            #     left = int(sp.root_scalar(func1,x0=0,x1=500).root)
            #     right = int(sp.root_scalar(fun2,x0=0,x1=500).root)
            #     ROI_mask[left:right + 1] = np.array(self.data[left:right + 1])
        if type == "bar":
            plt.bar(x=self.datax, height=ROI_mask , width=1, label = label,zorder = 2)#color="darkslateblue"
        elif type == "plot":
            plt.plot(self.datax,ROI_mask, label = label, zorder = 2)

    def calculate_centroid(self) -> list:
        centroid = []
        for ROI in self.ROI_limit:
            roi_poid = self.data[ROI[0]:ROI[1] + 1]
            centroid.append(round(sum(roi_poid * list(range(ROI[0], ROI[1] + 1))) / sum(roi_poid), 2))
        return centroid

    def remove_noise(self) -> object:
        self.data_noiseless = self.data.copy()
        noise_array = np.zeros(len(self.data))
        for index, ROI in enumerate(self.ROI_limit):
            x = self.get_frontier_points_x(index)
            y = self.get_frontier_points_y(index)
            noise_y = Noise(xdata=x, ydata=y)
            for i in range(ROI[0], ROI[1] + 1):
                noise_array[i] = noise_y.predict(i)
                self.data_noiseless[i] -= noise_y.predict(i)
                if self.data_noiseless[i] < 0:
                    self.data_noiseless[i] = 0
                    # noise_array[i] = self.data[i]
        return noise_array

    def remove_ROI(self, index: int) -> None:
        self.ROI_limit.pop(index)

    def get_frontier_points_y(self, ROInumber: int) -> list:
        frontier = self.ROI_limit[ROInumber]
        front = self.data[frontier[0] - 3:frontier[0] + 3]
        back = self.data[frontier[1] - 2:frontier[1] + 4]
        return np.concatenate((front, back), axis=None)

    def get_frontier_points_x(self, ROInumber: int) -> list:
        frontier = self.ROI_limit[ROInumber]
        front = self.datax[frontier[0] - 3:frontier[0] + 3]
        back = self.datax[frontier[1] - 2:frontier[1] + 4]
        return np.concatenate((front,back),axis = None)

    def add_ROI(self, leftlimit: int, rightlimit: int) -> None:
        self.ROI_limit.append((leftlimit, rightlimit))

    def get_FWHMs(self, plot=False) -> list:
        FWHMs = []
        for i in range(len(self.ROI_limit)):
            FWHM = abs(round(self.get_FWHM(i), 4))
            FWHMs.append(FWHM)
            peak = self.peaks[i]
            peakx = peak.mu
            peaksig = abs(peak.sigma)
            peaky = peak.gaussian_function(peak.mu) / 2
            mult = 2 if i % 2 == 0 else 4
            if plot:
                plt.annotate(" ", (peakx, peaky), xytext=(peakx + 50, mult * peaky / 3), arrowprops={'arrowstyle': '->'})
                plt.annotate("FWHM : {} [keV] \n Peak : {} [keV]".format(FWHM, round(peak.mucalib, 4)),
                            (peakx + 30, mult * peaky / 3), xytext=(peakx + 30, mult * peaky / 3), fontsize=6)
        return FWHMs

    def get_FWHM(self, index: int) -> float:
        peak = self.peaks[index]
        sig = peak.sigmacalib if self.calibrated else peak.sigma
        return (sig) * np.sqrt(2 * np.log(2))

    def plot_gaussian_over(self, index: int):
        peak = self.peaks[index]
        x = peak.x
        y = peak.y
        for index, xelem in enumerate(x):
            y[index] += self.noise_array[int(np.where(self.datax==int(xelem))[0][0])]
        plt.plot(x, y, "-", color="crimson", alpha=0.8)

    def plot_peaks(self):
        for i in range(len(self.ROI_limit)):
            self.plot_gaussian_over(i)

    def calculate_peak(self, ROI: tuple) -> object:
        x = self.datax[ROI[0]:ROI[1] + 1]
        y = self.data_noiseless[ROI[0]:ROI[1] + 1]
        return Gaussian(xdata=x, ydata=y)

    def calculate_peaks(self) -> None:
        self.noise_array = self.remove_noise()
        for index, ROI in enumerate(self.ROI_limit):
            if len(self.peaks) != len(self.ROI_limit):
                self.peaks.append(self.calculate_peak(ROI))
            else:
                self.peaks[index] = self.calculate_peak(ROI)

    def add_calibration(self,linalg:object):
        self.calibration = linalg
        self.calibrated = True

    def calibrate(self,energie = [], **kwargs) :
        echantillon = kwargs.get("echantillon","placeholder")
        plot = kwargs.get("plot",False)
        retourne = kwargs.get("retour",False)
        retour = None
        func = kwargs.get("func")
        if not self.calibrated:
            if not energie :
                raise Exception("L'énergie ne peut être une liste vide si on calibre")
            centroid = [0] + [self.peaks[i].mu for i in range(len(self.peaks))]
            linFit = Noise(centroid, [0] + energie,func=func)
            for index, peak in enumerate(self.peaks):
                peak.mucalib = linFit.predict(peak.mu)
                peak.sigma = linFit.predict(peak.sigma)
            self.calibrated = True
            self.calibration = linFit
            self.datax = self.calibration.predict(self.datax)
            # if plot:
            #     fig, ax = plt.subplots()
            #     ax.set_xlabel("Canaux")
            #     ax.set_ylabel("Énergie [keV]")
            #     # ax.set_title("Courbe de calibration {}".format(echantillon))
            #     ax.text(0.2, 0.8, "y = {}x + {}".format(round(linFit.param[0], 4), round(linFit.param[1], 4)),
            #             transform=ax.transAxes)
            #     ax.plot(self.datax, [yfunction(i) for i in range(len(self.datax))])
            #     ax.scatter(x=centroid, y=[0] + energie, marker="o", color="red")
            #     for index, center in enumerate(centroid[1:]):
            #         ax.annotate(energie[index], (center, energie[index]))
            #     fig.savefig("Courbe de calibration {}.png".format(echantillon), bbox_inches="tight")
            if retourne:
                retour = linFit

        else:
            for index, peak in enumerate(self.peaks):
                peak.mucalib = self.calibration.predict(peak.mu)
                peak.sigma = self.calibration.predict(peak.sigma)
                peak.incertitude = self.calibration.predict(peak.incertitude)
            self.datax = self.calibration.predict(self.datax)
            # for index,ROI in enumerate(self.ROI_limit):
            #     left = ROI[0]
            #     right = ROI[1]
            #     new_left = self.calibration.predict(left)
            #     new_right = self.calibration.predict(right)
            #     self.remove_ROI(index)
            #     self



        return retour

class Noise():
    def __init__(self, xdata: list, ydata: list,**kwargs):
        self.func = kwargs.get("func",self.func_lin)
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.param = sp.curve_fit(self.func,xdata = self.xdata,ydata = self.ydata)[0]

    def func_lin(self,x, a, b):
        return np.add(np.multiply(a,x),b)

    def predict(self,X,return_string = False) -> callable:
        return (self.func(X, *self.param),f"{self.param[0]:.4f}x ") if return_string else self.func(X, *self.param)


    def get_rsquared(self):
        res = self.ydata - self.predict(self.xdata)
        ss = np.sum(res**2)
        ss_tot = np.sum((self.ydata-np.mean(self.ydata))**2)
        return 1- (ss/ss_tot)

class Gaussian():
    def __init__(self, xdata: list, ydata: list):
        self.x = xdata
        self.datay = ydata
        param, pcov = sp.curve_fit(lambda x, mu, sigma, a: a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), xdata, ydata,
                                   p0=[np.mean(self.x), np.std(self.x), 1])
        self.mu = param[0]
        self.sigma = abs(param[1])
        self.mucalib = param[0]
        self.sigmacalib = abs(param[1])
        self.a = param[2]
        self.y = [self.gaussian_function(x) for x in self.x]
        self.incertitude = np.sqrt(np.diag(pcov))

    def gaussian_function(self, x) -> callable:
        return self.a * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))


# def Fait_toute_les_figures():
#     for filename in os.listdir(os.fsencode("Data")):
#         end = "AMDC" if filename.endswith(".mcs") else "maelstro"
#         spectre = BuildSpectre(filename, end)
#         spectre.show_Spectre(title="")
#         spectre.add_ROIS_to_fig(ax)
#         print(spectre.get_FWHMs())
#         for i in range(len(spectre.ROI_limit)):
#             spectre.plot_gaussian_over(ax, i)
#
#         spectre.save_fig(filename)


def sum_les_shit(files: list):
    rois = []
    Data = np.zeros(2048, int)
    for file in files:
        with open("Data/{}".format(file), "r") as datafile:
            rawstr = datafile.read().split("\n")
            ROI_index = rawstr.index("<<ROI>>")
            data_index = rawstr.index("<<DATA>>")
            config_index = rawstr.index("<<DPP CONFIGURATION>>")
            data_list = rawstr[data_index + 1:config_index - 1]
            ROI_list = rawstr[ROI_index + 1:data_index]
            CONFIG_list = rawstr[config_index + 1:]
            Data += np.array(data_list, int)
            rois += ROI_list
    with open("Data/SI-PIN.mca", "w") as outfile:
        outfile.write("<<ROI>>\n")
        for i in range(len(rois)):
            outfile.write(rois[i] + "\n")
        outfile.write("<<DATA>>\n")
        for i in range(len(Data)):
            outfile.write(str(Data[i]) + "\n")
        outfile.write("<<DPP CONFIGURATION>>")


def res_rel():
    names = ["Am Gain 20.90.mcs", "Am Gain 49.93.mcs", "Am Gain 76.70.mcs", "Am Gain 91.86.mcs"]
    gain = []
    res = []
    for name in names:
        spectre = BuildSpectre(name, "AMDC")
        spectre.calculate_peaks()
        pic = spectre.peaks[0].mu
        f = spectre.get_FWHM(0)
        gain.append(float(name[8:-4]))
        res.append(100 * f / pic)
    plt.scatter(x=gain, y=res, marker="^")
    plt.ylabel("Résolution relative [%]")
    plt.xlabel("Gain [dB]")
    plt.savefig("resolution rel")


def res_abs():
    names = ["Am Gain 20.90.mcs", "Am Gain 49.93.mcs", "Am Gain 76.70.mcs", "Am Gain 91.86.mcs"]
    gain = []
    res = []
    for name in names:
        spectre = BuildSpectre(name, "AMDC")
        spectre.calculate_peaks()
        pic = spectre.peaks[0].mu
        f = spectre.get_FWHM(0)
        gain.append(float(name[8:-4]))
        res.append(f)
    plt.scatter(x=gain, y=res, marker="^")
    plt.ylabel("Résolution absolue [canaux]")
    plt.xlabel("Gain [dB]")
    plt.savefig("resolution abs")


def res_vs_energy():
    energy = {"SI-PIN": [6.4, 13.95, 14.4, 22.1, 25, 59.54],
              "NaI": [122.06, 511, 661.66, 1173.228, 1332.492],
              "CdTe am": [13.95, 17.74, 59.54],
              "CdTe Cd": [22.1, 88],
              "CdTe Co": [14.4, 122]}
    fig, ax = plt.subplots()
    ax.set_ylabel("Résolution absolue [canaux]")
    ax.set_xlabel("Énergie des pics [keV]")
    ax.set(xlim=(0, 200))
    for file in os.listdir(os.fsencode("Data")):
        name = os.fsdecode(file)
        print(name)
        logiciel = "AMDC" if name.endswith(".mcs") else "maestro"
        spectre = BuildSpectre(name, logiciel)
        spectre.calculate_peaks()
        if name.startswith("SI-PIN"):
            energie = "SI-PIN"
            label = "Détecteur Si-PIN, toutes les sources, gain de 76,70 dB"
        elif name.startswith("Am"):
            energie = "CdTe am"
            label = "Détecteur CdTe, source Am-241,gain de 91,86 dB"
        elif name.startswith("Cd"):
            energie = "CdTe Cd"
            label = "Détecteur CdTe, source Cd-109, gain de 60,30 dB"
        elif name.startswith("Co gain41,59"):
            energie = "CdTe Co"
            label = "Détecteur CdTe, source Co-57, gain de 41,59 dB"
        else:
            energie = "NaI"
            label = "Détecteur NaI, toutes les sources, gain de 64 dB"
        print(energie)
        pics = spectre.peaks
        f = spectre.get_FWHMs()
        res_rel = [f[i] for i in range(len(f))]
        print(len(res_rel))
        scat = ax.scatter(x=energy[energie], y=res_rel, marker=".")
        scat.set_label(label)
        ax.legend(fancybox=True, framealpha=0.5)


class Data_base:
    def __init__(self, name: str, *args):
        self.db = {}
        for arg in args:
            self.db[arg] = {}
        self.name = name

    def __getitem__(self, item):
        return self.db[item]

    def __setitem__(self, key, value):
        self.db[key] = value

    @staticmethod
    def creat_database_activation():
        conversion = {"Ag":"Ag110","Ag2":"Ag108","Al":"Al28","calib":"Calibration","Cu":"Cu","V":"V52"}
        Data = Data_base("spectre_isotope","Calibration", "Ag108", "Ag110", "Al28", "V52", "Cu")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".Spe"):
                spectre = BuildSpectre(filename,"MAESTRO")
                temp_str = filename[:-4].split("_")
                title = conversion[temp_str[0]]
                sous_cat = "Spectre" if len(temp_str) == 1 else "Bruit"
                Data[title][sous_cat] = spectre
        return Data

    @staticmethod
    def create_dataBase_filt_x_courant_tension():
        DataBase = Data_base("variation_courant_tension", "Tension", "Courant")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".mca"):
                temp_str = filename[:-4].split("-")
                tension = float(temp_str[0][:-2])
                courant = float(temp_str[1][:-2])
                #filtre = temp_str[2]
                spectre = BuildSpectre(filename, "AMDC")
                temps = spectre.extradata["LIVE_TIME"]
                name = filename[:-4]
                cr_pack = {"Courant": courant, "Tension": tension, "Spectre": spectre, "Temps" : temps, "File name" : name}
                if tension not in DataBase["Tension"]:
                    DataBase["Tension"][tension] = [cr_pack]
                else:
                    DataBase["Tension"][tension].append(cr_pack)
                if courant not in DataBase["Courant"]:
                    DataBase["Courant"][courant] = [cr_pack]
                else:
                    DataBase["Courant"][courant].append(cr_pack)
        return DataBase

    @staticmethod
    def create_dataBase_filt_x_filtre_alter():
        DataBase = Data_base("filtres_alterne", "Filtre")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".mca"):
                temp_str = filename[:-4].split("-")
                if temp_str[0] != "nofilter":
                    filt1= temp_str[0]
                    filt2 = temp_str[1]
                    filtre = (filt1,filt2)
                else:
                    filtre = temp_str[0]
                spectre = BuildSpectre(filename, "AMDC")
                temps = spectre.extradata["LIVE_TIME"]
                name = filename[:-4]
                cr_pack = {"Filtre": filtre, "Spectre": spectre, "Temps": temps,"Filename": name}
                if filtre not in DataBase["Filtre"]:
                    DataBase["Filtre"][filtre] = [cr_pack]
                else:
                    DataBase["Filtre"][filtre].append(cr_pack)
        return DataBase
    @staticmethod
    def create_dataBase_filt_x_filtre():
        DataBase = Data_base("variation_filtres", "Filtre")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".mca"):
                temp_str = filename[:-4].split("-")
                tension = float(temp_str[0][:-2])
                courant = float(temp_str[1][:-2])
                filtre = temp_str[-1]
                spectre = BuildSpectre(filename, "AMDC")
                temps = spectre.extradata["LIVE_TIME"]
                name = filename[:-4]
                cr_pack = {"Courant": courant, "Tension": tension,"Filtre":filtre, "Spectre": spectre, "Temps": temps,
                           "Filename": name}
                if filtre not in DataBase["Filtre"]:
                    DataBase["Filtre"][filtre] = [cr_pack]
                else:
                    DataBase["Filtre"][filtre].append(cr_pack)
        return DataBase

    @staticmethod
    def creat_dataBase_stack_alu():
        DataBase = Data_base("stack_alu")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".mca"):
                epaisseur = filename[:-4]
                spectre = BuildSpectre(filename, "AMDC")
                temps = spectre.extradata["LIVE_TIME"]
                name = filename[:-4]
                cr_pack = {"Epaisseur": epaisseur, "Spectre": spectre, "Temps": temps,"Filename": name}
                DataBase[epaisseur] = [cr_pack]

        return DataBase

    @staticmethod
    def creat_dataBase_stack_cuivre():
        DataBase = Data_base("stack_cu")
        for file in os.listdir(os.fsencode("Data")):
            filename = os.fsdecode(file)
            if filename.endswith(".mca"):
                epaisseur = filename[:-4] if filename[:-4] != "0.1" else "0"
                spectre = BuildSpectre(filename, "AMDC")
                temps = spectre.extradata["LIVE_TIME"]
                name = filename[:-4]
                cr_pack = {"Epaisseur": epaisseur, "Spectre": spectre, "Temps": temps, "Filename": name}
                if epaisseur not in DataBase.db:
                    DataBase[epaisseur] = [cr_pack]
                else :
                    DataBase[epaisseur].append(cr_pack)
        return DataBase

    def save_DataBase(self):
        with open(f"{self.name}_data.pkl", "wb") as outfile:
            pickle.dump(self.db, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_database(name: str):
        with open(f"{name}.pkl","rb") as input:
            data = pickle.load(input)
            obj = Data_base(name)
            obj.db = data
        return obj
class Laboratoire_filt_x:
    def __init__(self):
        pass

    @staticmethod
    def plot_tension_constant():
        #Data_base.create_dataBase_filt_x_courant_tension().save_DataBase()
        data1 = Data_base.load_database("variation_courant_tension_data")
        courant_const = data1["Courant"]
        courant = list(courant_const.keys())
        courant.sort(reverse=True)
        list_fait = []
        list_emission_moyen = []
        list_centre_de_masse = []
        for elem in courant:
            objs = courant_const[elem]
            for index, obj in enumerate(objs):
                if obj["Tension"] == 30.0 and obj["Courant"] not in list_fait:
                    list_fait.append(obj["Courant"])
                    spectre = obj["Spectre"]
                    spectre.calibrate()
                    spectre.data = spectre.data / float(obj["Temps"])

                    canaux_actif = 0
                    mask = []
                    for index2,value in enumerate(spectre.data):
                        if value != 0:
                            canaux_actif += 1
                            mask.append(value)
                    list_emission_moyen = [np.sum(mask)] + list_emission_moyen
                    list_centre_de_masse = [np.average(list(range(len(spectre.data))),weights=spectre.data)] + list_centre_de_masse
                    spectre.show_Spectre(label=f"{elem} uA", type="plot",
                                         ylabel=r"Émission détectée par seconde [$s^{-1}$]")

        plt.legend(fancybox=True, framealpha=0.5)
        plt.annotate(f"{22.12} keV", xy=(22.12 , 1),  xytext = (23 , 1.5),arrowprops = {"arrowstyle":"->"})
        plt.annotate(f"{24.94} keV", xy=(24.94, 0.34), xytext=(25, 1.3), arrowprops={"arrowstyle": "->"})
        plt.xlim(0,40)
        plt.show()
        fonts = 25
        courant.sort()
        plt.ylabel(r"Détection par seconde [$s^{-1}$]",fontsize = fonts)
        plt.xlabel("Courant du générateur [uA]", fontsize=fonts)
        plt.grid()
        lin_reg1 = Noise(courant,list_emission_moyen)
        y1 = lin_reg1.predict(courant,True)
        plt.plot(courant,list_emission_moyen,"o",alpha = 0.6)
        plt.plot(courant,y1[0],"-",color = "red",alpha = 0.6)
        xtext = ((courant[int(len(y1[0]) / 2)] / courant[-1]) - 0.1) * courant[-1]
        plt.annotate(f"{y1[1]} \n" + rf"$R^2$ = {round(lin_reg1.get_rsquared(),3)}",xy = (xtext, lin_reg1.predict(xtext)),xytext = (0.5,0.85),textcoords = 'axes fraction',arrowprops = {"arrowstyle":"->"})
        plt.show()
        plt.xlabel("Courant du générateur [uA]",fontsize = fonts)
        plt.ylabel("Moyenne pondérée de l'activité des canaux [Canaux]",fontsize = fonts)
        plt.grid()
        lin_reg3 = Noise(courant, list_centre_de_masse)
        y3 = lin_reg3.predict(courant,True)
        plt.plot(courant,list_centre_de_masse,"o",alpha = 0.6)
        plt.plot(courant, y3[0], "-", color="red", alpha=0.6)
        plt.annotate(f"{y3[1]}\n" + rf"$R^2$ = {round(lin_reg3.get_rsquared(),3)}", xy=(xtext, lin_reg3.predict(xtext)), xytext=(0.5, 0.75), textcoords='axes fraction',
                         arrowprops={"arrowstyle": "->"})

        plt.show()

    @staticmethod
    def plot_courant_constant():
        #Data_base.create_dataBase_filt_x_courant_tension().save_DataBase()
        data1 = Data_base.load_database("variation_courant_tension_data")
        courant_const = data1["Tension"]
        courant = list(courant_const.keys())
        courant.sort(reverse=True)
        list_fait = []
        list_emission_moyen = []
        list_centre_de_masse = []
        for elem in courant:
            objs = courant_const[elem]
            for index, obj in enumerate(objs):
                if obj["Courant"] == 12.0 and obj["Tension"] not in list_fait:
                    list_fait.append(obj["Tension"])
                    spectre = obj["Spectre"]
                    spectre.calibrate()
                    spectre.data = spectre.data / float(obj["Temps"])
                    canaux_actif = 0
                    mask = []
                    for index2, value in enumerate(spectre.data):
                        if value != 0:
                            canaux_actif += 1
                            mask.append(value)
                    list_emission_moyen = [np.sum(mask)] + list_emission_moyen
                    list_centre_de_masse = [np.average(list(range(len(spectre.data))),
                                                       weights=spectre.data)] + list_centre_de_masse
                    spectre.show_Spectre(label=f"{elem} kV", type="plot",
                                         ylabel=r"Émission détectée par seconde [$s^{-1}$]")

        plt.legend(fancybox=True, framealpha=0.5)
        plt.annotate(f"{22.12} keV", xy=(22.12, 1), xytext=(30, 1.5), arrowprops={"arrowstyle": "->"})
        plt.annotate(f"{24.94}keV", xy=(24.94, 0.34), xytext=(34, 1.3), arrowprops={"arrowstyle": "->"})
        plt.xlim(0, 40)
        plt.show()
        fonts = 25
        courant.sort()
        plt.ylabel(r"Détection par seconde [$s^{-1}$]", fontsize=fonts)
        plt.xlabel("Tension du générateur [kV]", fontsize=fonts)
        plt.grid()
        quad = lambda x, a, b,c: a * x ** 2 + b * x + c
        param, pcov = sp.curve_fit(quad, courant, list_emission_moyen)
        quad_fit = lambda x: param[0] * x ** 2 + param[1] * x + param[2]
        new_x = np.linspace(courant[0],courant[-1],50)
        y1 = quad_fit(new_x)
        plt.plot(courant, list_emission_moyen, "o", alpha=0.6)
        plt.plot(new_x, y1, "-", color="red", alpha=0.6)
        xtext = ((new_x[int(len(y1) / 2)] / new_x[-1]) - 0.1) * new_x[-1]
        plt.annotate(fr"{param[0]:.4}$x^2$ + {param[1]:.4}x + {param[2]:.4f}",
                     xy=(xtext, quad_fit(xtext)), xytext=(0.2, 0.75), textcoords='axes fraction',
                     arrowprops={"arrowstyle": "->"},fontsize = 20)
        plt.show()
        plt.xlabel("Tension du générateur [kV]", fontsize=fonts)
        plt.ylabel("Moyenne des canaux, pondérée par leur activité [Canaux]", fontsize=20)
        plt.grid()

        quad = lambda x, a, b, c : a*x**2 + b*x + c
        param,pcov = sp.curve_fit(quad,courant,list_centre_de_masse)
        quad_fit = lambda x : param[0]*x**2 + param[1]*x + param[2]
        new_x = np.linspace(courant[0], courant[-1], 50)
        y1 = quad_fit(new_x)
        plt.plot(courant, list_centre_de_masse, "o", alpha=0.6)
        plt.plot(new_x, y1, "-", color="red", alpha=0.6)
        xtext = ((new_x[int(len(y1) / 2)] / new_x[-1]) - 0.1) * new_x[-1]
        plt.annotate(fr"{param[0]:.4}$x^2$ + {param[1]:.4}x + {param[2]:.4f}",
                     xy=(xtext, quad_fit(xtext)), xytext=(0.5, 0.75), textcoords='axes fraction',
                     arrowprops={"arrowstyle": "->"}, fontsize=20)

        plt.show()

    @staticmethod
    def plot_spectre_filtre():
        data1 = Data_base.create_dataBase_filt_x_filtre()
        # data1 = Data_base.load_database("variation_filtres_data")
        filtres = data1["Filtre"]
        nofiltre_spectre = data1["Filtre"]["nofilter"][0]["Spectre"]
        nofiltre_spectre.data = nofiltre_spectre.data/ float(data1["Filtre"]["nofilter"][0]["Temps"])
        nofiltre_spectre.calibrate()
        centro = 31
        right_lim = centro + 0.02052
        left_lim = centro - 0.02052
        dict = {"filtre":{},"Z":{}}
        centre = 1926
        ind3 = np.where((nofiltre_spectre.datax < 20) & (nofiltre_spectre.datax > 10))[0]
        temp2 = [nofiltre_spectre.data[i] for i in ind3]
        dict["filtre"]["nofilter"] = {"Count": nofiltre_spectre.data[centre],"Bande" :sum(temp2)}
        for filtre in filtres:
            spectre = filtres[filtre][0]["Spectre"]
            if filtre != "nofilter":

                spectre.data = spectre.data / float(filtres[filtre][0]["Temps"])
                spectre.calibrate()
                spectre.show_Spectre(type = "plot",label = f"Spectre du mini-X avec filtre de {filtre}")
                nofiltre_spectre.show_Spectre(type = "plot",label = "Spectre du mini-X sans filtre",
                                              ylabel=r"Émission détectée par seconde [$s^{-1}$]")
                plt.legend(fancybox=True, framealpha=0.5,fontsize = 20)
                plt.xlim(0,40)
                temp = spectre.data[centre]
                temp2 = [spectre.data[i] for i in ind3]
                dict["filtre"][filtre] = {"Count":temp,"Bande" :sum(temp2)}
        plt.show()
        for filtre in dict["filtre"]:
            N_a = 6.02214076e23
            if filtre != "nofilter":
                count = dict["filtre"][filtre]["Count"]
                if filtre == "Aluminium":
                    dict["filtre"][filtre]["A"] = 27
                    dict["filtre"][filtre]["Z"] = 13
                    dict["filtre"][filtre]["p"] = 2.7*10*0.00254

                if filtre == "Argent":
                    dict["filtre"][filtre]["A"] = 108
                    dict["filtre"][filtre]["Z"] = 47
                    dict["filtre"][filtre]["p"] = 10.49*0.00254
                if filtre == "Cuivre":
                    dict["filtre"][filtre]["A"] = 63
                    dict["filtre"][filtre]["Z"] = 29
                    dict["filtre"][filtre]["p"] = 8.92*0.00254
                if filtre == "Molybdène":
                    dict["filtre"][filtre]["A"] = 96
                    dict["filtre"][filtre]["Z"] = 42
                    dict["filtre"][filtre]["p"] = 10.28*0.00254
                if filtre == "Tungstène":
                    dict["filtre"][filtre]["A"] = 184
                    dict["filtre"][filtre]["Z"] = 74
                    dict["filtre"][filtre]["p"] = 19.25*0.00254

                ln = -np.log(count/dict["filtre"]["nofilter"]["Count"])

                pt = dict["filtre"][filtre]["p"]

                A = dict["filtre"][filtre]["A"]/N_a

                a_tau = ((ln/pt)-0.2)*(A)

                Z = dict["filtre"][filtre]["Z"]
                dict["Z"][Z] = a_tau
        z = list(dict["Z"].keys())
        z.sort()
        a = []
        for elem in z:
            a.append(dict["Z"][elem])
        z = np.array(z)
        a = np.array(a)
        linfit = Noise(xdata=np.log(z), ydata=np.log(a))
        param = linfit.param
        plt.annotate(f"{param[0]:.3}x  {param[1]:.3}\n"+ rf"$R^2$ = {linfit.get_rsquared():.3}",(0.2,0.5),xycoords  = "axes fraction")
        plt.plot(np.log(z),np.log(a),"o")
        plt.plot(np.log(z),linfit.predict(np.log(z)),"-",color = "r")
        plt.ylabel(r"$\ln{_a\tau}$ [-]")
        plt.xlabel(r"$\ln{Z}$ [-]")
        plt.show()


    @staticmethod
    def plot_spectre_filtre_alt():
        data1 = Data_base.load_database("filtres_alterne_data")
        filtres = data1["Filtre"]
        list_fait = ["nofilter"]
        centro = 17
        nofilter = filtres["nofilter"][0]
        nofilter_s = filtres["nofilter"][0]["Spectre"]
        nofilter_s.data = nofilter_s.data / float(nofilter["Temps"])
        nofilter_s.calibrate()
        right_lim = centro + 0.02052  # 22.14052
        left_lim = centro - 0.02052
        for filtre in filtres:
            obj = filtres[filtre][0]
            filtre_tuple = filtre
            if filtre_tuple not in list_fait and (filtre_tuple[1],filtre_tuple[0]) not in list_fait:
                list_fait.append(filtre_tuple)
                spectre = obj["Spectre"]
                spectre.data = spectre.data / float(obj["Temps"])
                spectre.calibrate()
                obj_inverse = filtres[(filtre[1],filtre[0])][0]
                spectre_inv = obj_inverse["Spectre"]
                spectre_inv.data = spectre_inv.data / float(obj["Temps"])
                spectre_inv.calibrate()
                spectre.show_Spectre(type = "plot",label = f"Filtre de {filtre[0]} et filtre {filtre[1]}")
                spectre_inv.show_Spectre(type = "plot",label = f"Filtre de {filtre[1]} et filtre {filtre[0]}",
                                              ylabel=r"Émission détectée par seconde [$s^{-1}$]")
                nofilter_s.show_Spectre(type="plot", label=f"Sans filtre")
                ind1 = np.where((spectre.datax <  22.3) & (spectre.datax >  22.0))
                ind2 = np.where((spectre.datax <  25.0) & (spectre.datax >  24.5))
                ind1 = int(ind1[0][0])
                ind2 = int(ind2[0][-1])
                y = [spectre.data[ind1],spectre.data[ind2]]
                x1 = 30/spectre.datax[-1]
                x2 = 34/spectre.datax[-1]
                plt.legend(fancybox = True, framealpha = 0.5,fontsize = 20)
                plt.xlim(0,40)
                plt.show()

    @staticmethod
    def plot_spectre_stack_alu():
        data1 = Data_base.load_database("stack_alu_data")
        clef = [int(i) for i in data1.db.keys()]
        clef.sort()
        list_count = []
        centro = 17
        right_lim = centro + 0.02052  # 22.14052
        left_lim = centro - 0.02052  # 22.09948
        list_count_larg = []
        d,g = 20,10
        for filtre in clef:
            obj = data1[str(filtre)][0]
            spectre = obj["Spectre"]
            spectre.data = spectre.data / float(obj["Temps"])
            spectre.calibrate()
            label = f"Filtre d'aluminium de {filtre} mils" if filtre != 0 else "Sans filtre"
            if filtre != 1:
                spectre.show_Spectre(type="plot",
                                     label=label,ylabel=r"Émission détectée par seconde [$s^{-1}$]")
            ind1 = np.where((spectre.datax < 22.3) & (spectre.datax > 22.0))
            ind2 = np.where((spectre.datax < 25.0) & (spectre.datax > 24.5))
            ind3 = np.where((spectre.datax < right_lim) & (spectre.datax > left_lim))[0]
            ind4 = np.where((spectre.datax < d) & (spectre.datax > g))[0]
            centre = int(ind3[int(len(ind3)/2)])
            temp = spectre.data[centre]
            temp2 = [spectre.data[i] for i in ind4]
            list_count_larg.append(sum(temp2))
            list_count.append(temp)
            ind1 = int(ind1[0][0])
            ind2 = int(ind2[0][-1])
            y = [spectre.data[ind1], spectre.data[ind2]]
            x1 = 30 / spectre.datax[-1]
            x2 = 34 / spectre.datax[-1]

        plt.legend(fancybox=True, framealpha=0.5,fontsize = 18)
        plt.xlim(0, 40)
        #plt.tight_layout()
        plt.show()
        y = []
        x = []
        y_2 = []
        for index,val in enumerate(list_count):
            if index != 1:
                if int(clef[index]) in [70,80,90,100]:
                    y.append(val/list_count[1])
                    y_2.append(list_count_larg[index]/list_count_larg[1])
                    #y[-1] += 0.2
                    x.append(clef[index]* 0.00254)
                else:
                    y.append(val/list_count[0])
                    y_2.append(list_count_larg[index] / list_count_larg[0])
                    x.append(clef[index]* 0.00254)
        plt.plot(x,y,"o",label = f"Canal mono-énergétique à {centro} keV")

        new_x = np.array(x)# x[0:2]+x[3:6]+x[7:10]
        new_y = np.array(y) # y[0:2]+y[3:6]+y[7:10]
        param,pcov = sp.curve_fit(lambda x, u: np.exp(-u*x) , xdata=new_x, ydata=new_y)
        func = lambda x: np.exp(-np.multiply(x,param[0]))
        func_inv = lambda y: np.log(y)/(-1*param[0])
        list_x = np.linspace(x[0], x[-1], 50)
        plt.plot(list_x, func((np.array(list_x))), "-", color="r")
        demi = func_inv(0.5)
        quart = func_inv(0.25)
        plt.plot(x, [0.25 for i in range(len(x))], "--")
        plt.plot(x, [0.5 for i in range(len(x))], "--")
        plt.plot([demi,quart],[0.5,0.25],"o",color = "purple")
        sigma = np.sqrt(np.diag(pcov)[0])
        plt.annotate(fr"$CDA_{{1}}$ à t = {demi:.3} cm",(demi,0.51),fontsize = 20)
        plt.annotate(fr"$CDA_{{2}}$ à t = {quart:.3} cm",(quart,0.26),fontsize = 20)
        plt.annotate(fr"$\frac{{N_t}}{{N_0}} = e^{{-({param[0]:.0f}\pm {3*sigma:.0f})t}}$", xy=(list_x[int(len(list_x)/2)],func(list_x[int(len(list_x)/2)])), xytext=(0.5, 0.6), textcoords = "axes fraction", arrowprops={"arrowstyle": "->"})
        plt.xlabel("Épaisseur du filtre d'aluminium [cm]")
        plt.ylabel(rf"$\frac{{N_t}}{{N_0}}$ à {centro} keV [-]")
        plt.xlim(0,0.26)
        plt.show()
        plt.plot(x, y_2, "o", label=f"Canaux entre {g} et {d} keV")
        param, pcov = sp.curve_fit(lambda x, u: np.exp(-u * x), xdata=new_x, ydata=y_2)
        func = lambda x: np.exp(-np.multiply(x, param[0]))
        func_inv = lambda y: np.log(y) / (-1 * param[0])
        list_x = np.linspace(x[0], x[-1], 50)
        plt.plot(list_x, func((np.array(list_x))), "-", color="r")

        res = y_2 - func(x)
        ss = np.sum(res ** 2)
        ss_tot = np.sum((y_2 - np.mean(y_2)) ** 2)
        rsquared = 1 - (ss / ss_tot)

        demi = func_inv(0.5)
        quart = func_inv(0.25)
        plt.plot(x, [0.25 for i in range(len(x))], "--")
        plt.plot(x, [0.5 for i in range(len(x))], "--")
        plt.plot([demi, quart], [0.5, 0.25], "o", color="purple")
        sigma = np.sqrt(np.diag(pcov)[0])
        plt.annotate(fr"$CDA_{{1}}$ à t = {demi:.3} cm", (demi, 0.51), fontsize=20)
        plt.annotate(fr"$CDA_{{2}}$ à t = {quart:.3} cm", (quart, 0.26), fontsize=20)
        plt.annotate(fr"$\frac{{N_t}}{{N_0}} = e^{{-({param[0]:.0f}\pm {3 * sigma:.0f})t}}$" + "\n"+ rf"$R^2 = {rsquared:.3}$",
                     xy=(list_x[int(len(list_x) / 2)], func(list_x[int(len(list_x) / 2)])), xytext=(0.5, 0.6),
                     textcoords="axes fraction", arrowprops={"arrowstyle": "->"})
        plt.xlabel("Épaisseur du filtre d'aluminium [cm]")
        plt.ylabel(rf"$\frac{{N_t}}{{N_0}}$ entre {g} keV et {d} keV [-]")
        plt.xlim(0, x[-1])
        plt.show()

    @staticmethod
    def plot_spectre_stack_cu():
        data1 = Data_base.load_database("stack_cu_data")
        clef = [int(i) for i in data1.db.keys()]
        clef.sort()
        list_count = []
        centro = 17
        right_lim = centro + 0.02052  # 22.14052
        left_lim = centro - 0.02052  # 22.09948
        for filtre in clef:
            obj = data1[str(filtre)][0]
            spectre = obj["Spectre"]
            spectre.data = spectre.data / float(obj["Temps"])
            spectre.calibrate()
            if filtre == 0:
                label = "Sans filtre"
                obj2 = data1[str(filtre)][1]
                spectre2 = obj2["Spectre"]
                spectre2.data = spectre2.data / float(obj2["Temps"])
                spectre2.calibrate()
                ind3 = np.where((spectre.datax < right_lim) & (spectre.datax > left_lim))[0]
                centre = int(ind3[int(len(ind3) / 2)])
                temp = spectre.data[centre]
                list_count.append(temp)

            elif filtre == 1:
                label = f"Filtre de cuivre de {filtre} mil"
            else:
                label = f"Filtre de cuivre de {filtre} mils"

            spectre.show_Spectre(type="plot",label=label,ylabel=r"Émission détectée par seconde [$s^{-1}$]")
            ind1 = np.where((spectre.datax < 22.3) & (spectre.datax > 22.0))
            ind2 = np.where((spectre.datax < 25.0) & (spectre.datax > 24.5))
            ind3 = np.where((spectre.datax < right_lim) & (spectre.datax > left_lim))[0]
            centre = int(ind3[int(len(ind3) / 2)])
            temp = spectre.data[centre]
            list_count.append(temp)
            ind1 = int(ind1[0][0])
            ind2 = int(ind2[0][-1])
            y = [spectre.data[ind1], spectre.data[ind2]]
            x1 = 30 / spectre.datax[-1]
            x2 = 34 / spectre.datax[-1]
        plt.legend(fancybox=True, framealpha=0.5,fontsize = 18)
        plt.xlim(0,40)
        # plt.tight_layout()
        plt.show()

        y = []
        x = []
        for index,val in enumerate(list_count[1:]):
            if clef[index] == 1:
                y.append(val/list_count[0])
                #y[-1] += 0.2
                x.append(clef[index] * 0.00254)
            else:
                y.append(val/list_count[1])
                x.append(clef[index]* 0.00254)
        plt.plot(x,y,"o")
        new_x = x  # x[0:2]+x[3:6]+x[7:10]
        new_y = y  # y[0:2]+y[3:6]+y[7:10]
        param, pcov = sp.curve_fit(lambda x, u: np.exp(-u * x), xdata=new_x, ydata=new_y)
        sigma = np.sqrt(np.diag(pcov)[0])
        func = lambda x: np.exp(-np.multiply(x, param[0]))
        func_inv = lambda y: np.log(y) / (-1 * param[0])
        demi = func_inv(0.5)
        quart = func_inv(0.25)
        list_x = np.linspace(x[0],x[-1],50)
        plt.plot(list_x, func((np.array(list_x))), "-", color="r")
        plt.plot( x, [0.25 for i in range(len(x))], "--")
        plt.plot( x, [0.5 for i in range(len(x))], "--")
        plt.plot(demi, 0.5, "o")
        plt.plot(quart, 0.25, "o")

        plt.annotate(fr"$CDA_{{1}}$ à t = {demi:.3} cm", (demi, 0.51), fontsize=18)
        plt.annotate(fr"$CDA_{{2}}$ à t = {quart:.3} cm", (quart, 0.26), fontsize=18)
        plt.annotate(fr"$\frac{{N_t}}{{N_0}} = e^{{-({param[0]:.0f}\pm {3*sigma:.0f})t}}$", xy=(list_x[int(len(list_x)/2)],func(list_x[int(len(list_x)/2)])), xytext=(0.5, 0.8), textcoords = "axes fraction", arrowprops={"arrowstyle": "->"})
        plt.xlabel("Épaisseur du filtre de cuivre [cm]")
        plt.ylabel(rf"$\frac{{N_t}}{{N_0}}$ (à {centro} keV) [-]")
        plt.xlim(0,x[-1])
        plt.show()

class Laboratoire_activation:
    def __init__(self):
        self.datab = self.load_db()

    def load_db(self):
        return Data_base.load_database("spectre_isotope_data")

    @staticmethod
    def enelever_bruit(spectre:BuildSpectre,bruit:BuildSpectre):
        for i in range(len(spectre.data)):
            temp = spectre.data[i] - bruit.data[i]
            if temp < 0:
                temp = 0

            spectre.data[i] = temp

    def calibrer(self)->Noise:
        energie = [122,662,1173,1333]
        calibration:BuildSpectre = self.datab["Calibration"]["Spectre"]
        calibration.calculate_peaks()
        retour = calibration.calibrate(energie,retour=True,func = lambda x, a: np.multiply(a,x))
        # calibration.show_Spectre()
        # plt.xlim(0, calibration.datax[-1])
        # plt.show()
        return retour

    def prod_spectre(self):
        calibration = self.calibrer()
        keys = list(self.datab.db.keys())
        cu = keys.pop(keys.index("Cu"))
        keys= [cu]+keys
        for key in keys:
            iso = self.datab[key]
            spectre:BuildSpectre = iso["Spectre"]
            spectre.data = spectre.data/float(spectre.extradata.get("LIVE_TIME"))
            spectre.add_calibration(calibration)

            if key != "Calibration" :
                bruit = iso["Bruit"]
                bruit.data = bruit.data / float(bruit.extradata.get("LIVE_TIME"))
                self.enelever_bruit(spectre,bruit)
                spectre.add_calibration(calibration)

                if key == "Ag108":
                    fun1 = lambda x : calibration.func_parametre(x) - 580
                    fun2 = lambda x: calibration.func_parametre(x) - 693
                    left = int(sp.root_scalar(fun1,bracket = [0,2000]).root)
                    right = int(sp.root_scalar(fun2, bracket=[0, 2000]).root)
                    spectre.add_ROI(left,right)
                    spectre.calculate_peaks()
                    spectre.calibrate()

                elif key == "Ag110":
                    fun1 = lambda x : calibration.func_parametre(x) - 620
                    fun2 = lambda x: calibration.func_parametre(x) - 690
                    left = int(sp.root_scalar(fun1,bracket = [0,2000]).root)
                    right = int(sp.root_scalar(fun2, bracket=[0, 2000]).root)
                    spectre.add_ROI(left,right)
                    spectre.calculate_peaks()
                    spectre.calibrate()
                else:
                    spectre.calculate_peaks()
                    spectre.calibrate()
                if key == "Cu":
                    spectre.show_Spectre(type="plot", color="grey")

                spectre.add_ROIS_to_fig(label = key,type = "plot")
                spectre.annotate_centroid()

        plt.xlabel("Énergie [keV]")
        plt.ylabel("Émission détecté par seconde [comptes/s]")
        plt.legend(fancybox = True, framealpha = 0.5)
        plt.show()

    @staticmethod
    def calculer_K():
        dict_elem = {"Al":{"G":0.40,"K_t":[7/100],"K_p":0.13,"f":1},
                     "Cu":{"G":0.27,"K_t":[7/100],"K_p":0.15,"f":0.09},
                     "V":{"G":0.27,"K_t":[7/100],"K_p":0.15,"f":1}}

        retour = {}
        for elem in dict_elem:
            # print(elem)
            mini_dic = dict_elem[elem]
            K = mini_dic["G"]*mini_dic["K_t"][-1] * mini_dic["K_p"] * mini_dic["f"]
            # print(f"{K:.4}")
            retour[f"{elem}"] = K
        return retour

    @staticmethod
    def calculer_lambda():
        dict_elem = {"Al": {"1/2 life":2.241*60},
                     "Cu": {"1/2 life":2.37*60},
                     "V": {"1/2 life":3.743*60}}
        const = 0.693147
        retour = {}
        for elem in dict_elem:
            retour[elem] = const/dict_elem[elem]["1/2 life"]
        return retour

    def calculate_activite_init(self):
        dict_elem = {"Al": {"live": 425,"Spectre":self.datab["Al28"]["Spectre"],"Bruit":self.datab["Al28"]["Bruit"]},
                     "Cu": {"live": 302,"Spectre":self.datab["Cu"]["Spectre"],"Bruit":self.datab["Cu"]["Bruit"]},
                     "V":{"live": 600,"Spectre":self.datab["V52"]["Spectre"],"Bruit":self.datab["V52"]["Bruit"]}}
        lambdas = self.calculer_lambda()
        ks = self.calculer_K()
        retour = {}
        for elem in dict_elem:
            donne = dict_elem[elem]
            spectre = donne["Spectre"]
            bruit = donne["Bruit"]
            bruit.data = bruit.data / float(bruit.extradata.get("LIVE_TIME"))
            spectre.data = spectre.data / float(spectre.extradata.get("LIVE_TIME"))

            k = ks[elem]
            y = lambdas[elem]
            self.enelever_bruit(spectre,bruit)
            gamma = self.get_ROI_data(spectre)
            num =  gamma * y
            exp1 = np.exp(-y*10)
            exp2 = 1 - np.exp(-y * donne["live"])
            denu = k * exp1 * exp2
            retour[elem] = num/denu
        return retour

    def calculate_flux(self):
        N_a = 6.023e23
        dict_elem = {
            "Al": {"temps_activation": 7.86e3, "alpha":1,"m":159.367,"sigma":0.23*10**(-24),"W":27},
            "Cu": {"temps_activation": 1.674e4, "alpha":0.3083,"m":336.048459,"sigma":2.17*10**(-24),"W":65}}
        As = self.calculate_activite_init()
        lambdas = self.calculer_lambda()
        retour = {}
        for elem in dict_elem:
            donne = dict_elem[elem]
            y = lambdas[elem]
            N_c = donne["alpha"] * donne["m"] * N_a * (donne["W"])**(-1)
            print(f"{elem} :{N_c}")
            A = As[elem]
            S = 1 - np.exp(-y*donne["temps_activation"])
            retour[elem] = A/(donne["sigma"] * N_c * S)
        return retour

    def get_ROI_data(self,spectre:BuildSpectre):
        return np.sum(spectre.data)

    def calcul_section_efficace(self):
        flux = [3.53e3,1.41e4]
        N_a = 6.023e23
        dict_elem = {
            "V": {"temps_activation": 4.5e3, "alpha": 0.9975, "m": 42.32, "W": 51},
            "Cu": {"temps_activation": 1.674e4, "alpha": 0.3083, "m": 336.048459, "W": 65}}
        As = self.calculate_activite_init()
        retour = {}
        for elem in dict_elem:
            donne = dict_elem[elem]
            N_c = donne["alpha"] * donne["m"] * N_a * (donne["W"])**(-1)
            section = []
            for f in flux:
                section.append(As[elem]/(f * N_c * 1))
            retour[elem] = section
        return retour






if __name__ == "__main__":
    Data_base.create_dataBase_filt_x_filtre_alter().save_DataBase()
    Laboratoire_filt_x.plot_spectre_filtre_alt()
    # lab = Laboratoire_activation()
    # fluxes = lab.calcul_section_efficace()
    # act = lab.calculate_activite_init()
    # for elem in fluxes:
    #     print(elem)
    #     print(f"activité : {act[elem]:.4}")
    #     for index,i in enumerate(fluxes[elem]):
    #         if index == 1:
    #             print(f"Section efficace pour flux max: {i*10**(24):.4}")
    #         else:
    #             print(f"Section efficace pour flux min: {i * 10 ** (24):.4}")
    #lab.calculer_flux()

    # data1 = Data_base.load_database("variation_courant_tension_data")
    # courant_const = data1["Courant"]
    # spectre = courant_const[17.0][0]["Spectre"]
    # spectre.add_ROI(1315,1400)
    # spectre.add_ROI(1500, 1580)
    # spectre.calculate_peaks()
    # spectre.calibrate()
    # print([spectre.calibration.predict(i.mu) for i in spectre.peaks])

    # spectre1 = BuildSpectre(name,"AMDC")
    # CdTe AM
    # energy = [13.95,59.54]
    # Si-PIN
    # energy = [6.4,13.95,14.4,22.1,25,59.54]
    # NaI
    # energy = [122.06, 511, 661.66, 1173.228, 1332.492]
    # spectre1.calculate_peaks()
    # spectre1.calibrate(energie=energy, echantillon=name[:2],plot=True)
    # fig, ax = spectre1.show_Spectre("")
    # spectre1.add_ROIS_to_fig(ax)
    # spectre1.get_FWHMs(ax)
    # for i in range(len(spectre1.ROI_limit)):
    #    spectre1.plot_gaussian_over(ax,i)
    # plt.tight_layout()
    # spectre1.save_fig(name[:-4]+".png")
    # plt.savefig("energievsresa1")
