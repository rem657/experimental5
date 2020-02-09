import matplotlib.pyplot as plt
import  numpy as np
import scipy.optimize as sp
import os

class BuildSpectre():
    def __init__(self,fileName,type):
        self.calibrated = False
        self.extradata = None
        if type == "maelstro" :
            self.data,ROIs  =  self.__importData_gamma(fileName)
        if type == "AMDC":
             self.data,ROIs,self.extradata = self.__importData_xray(fileName)
        else:
            assert AssertionError(" Le type est inconnue")
        self.ROI_limit = []
        self.peaks = []
        self.datax = list(range(0,len(self.data)))
        self.data_noiseless = self.data.copy()
        for coord in ROIs:
            coords = coord.split(" ")
            self.ROI_limit.append((int(coords[0]),int(coords[1])))
        self.noise_array = np.zeros(len(self.data))
    @staticmethod
    def __importData_gamma(fileName:str)-> tuple:
        with open("Data/{}".format(fileName),"r") as datafile:
            rawstr = datafile.read().split("\n")
            ROI_index = rawstr.index("$ROI:")
            data_list =  rawstr[12:ROI_index]
            ROI_bloc = rawstr[ROI_index+2:rawstr.index("$PRESETS:")]
        return np.array(data_list,dtype="int"),ROI_bloc

    @staticmethod
    def __importData_xray(fileName:str)-> tuple:
        with open("Data/{}".format(fileName),"r") as datafile:
            rawstr = datafile.read().split("\n")
            ROI_index = rawstr.index("<<ROI>>")
            data_index = rawstr.index("<<DATA>>")
            config_index = rawstr.index("<<DPP CONFIGURATION>>")
            data_list =  rawstr[data_index+1:config_index-1]
            ROI_list = rawstr[ROI_index+1:data_index]
            CONFIG_list = rawstr[config_index+1:]

        return np.array(data_list,dtype="int"),ROI_list,CONFIG_list

    def show_Spectre(self,title="placeHolder")->tuple:
        fig,ax = plt.subplots()
        data = self.data
        ax.bar(x = self.datax,height = data,width=1)
        ax.set_xlabel("Canaux")
        ax.set_ylabel("Nombre d'impulsions détectées")
        ax.set_title("{}".format(title))
        plt.margins(x=0,y=0)
        return fig,ax

    def save_fig(self,figureName:str)->None:
        plt.savefig(figureName)

    def add_ROIS_to_fig(self,ax):
        ROI_mask = np.zeros(len(self.data), dtype=int)

        for coords in self.ROI_limit:
            ROI_mask[coords[0]:coords[1]+1] = self.data[coords[0]:coords[1]+1]
        ax.bar(x = self.datax,height=ROI_mask,color = "darkslateblue",width = 1)

    def calculate_centroid(self)->list:
        centroid = []
        for ROI in self.ROI_limit:
            roi_poid = self.data[ROI[0]:ROI[1]+1]
            centroid.append(round(sum(roi_poid*list(range(ROI[0],ROI[1]+1)))/sum(roi_poid),2))
        return centroid

    def remove_noise(self)->object:
        self.data_noiseless = self.data.copy()
        noise_array = np.zeros(len(self.data))
        for index,ROI in enumerate(self.ROI_limit):
            x = self.get_frontier_points_x(index)
            y = self.get_frontier_points_y(index)
            noise_y = Noise(xdata= x, ydata=y).get_noise()
            for i in range(ROI[0],ROI[1]+1):
                noise_array[i]=noise_y(i)
                self.data_noiseless[i] -= noise_y(i)
                if self.data_noiseless[i] < 0:
                    self.data_noiseless[i] =0
                    #noise_array[i] = self.data[i]
        return noise_array

    def remove_ROI(self,index:int)->None:
        self.ROI_limit.pop(index)

    def get_frontier_points_y(self,ROInumber:int)->list:
        frontier = self.ROI_limit[ROInumber]
        front = self.data[frontier[0] - 3:frontier[0] + 3]
        back = self.data[frontier[1] - 2:frontier[1] + 4]
        return np.concatenate((front,back),axis=None)

    def get_frontier_points_x(self,ROInumber:int)->list:
        frontier = self.ROI_limit[ROInumber]
        front = self.datax[frontier[0]-3:frontier[0]+3]
        back = self.datax[frontier[1]-2:frontier[1]+4]
        return front + back

    def add_ROI(self,leftlimit:int,rightlimit:int)->None:
        self.ROI_limit.append((leftlimit,rightlimit))

    def get_FWHMs(self,ax)->list:
        FWHMs = []
        for i in range(len(self.ROI_limit)):
            FWHM = abs(round(self.get_FWHM(i),4))
            FWHMs.append(FWHM)
            peak = self.peaks[i]
            peakx = peak.mu
            peaksig = abs(peak.sigma)
            peaky = peak.gaussian_function(peak.mu) / 2
            ax.annotate(" ", (peakx, peaky),xytext = (peakx+50,4*peaky/3),arrowprops={'arrowstyle': '->'})
            ax.annotate("FWHM : {} [keV] \n Peak : {} [keV]".format(FWHM,round(peak.mucalib,4)),(peakx+50,4*peaky/3),xytext = (peakx+2*(FWHM),4*peaky/3),fontsize = 6)
        return FWHMs

    def get_FWHM(self,index:int)->float:
        peak = self.peaks[index]
        return (peak.sigma)*np.sqrt(2*np.log(2))

    def plot_gaussian_over(self,ax,index:int):
        peak = self.peaks[index]
        x = peak.x
        y = peak.y
        for index,xelem in enumerate(x):
            y[index] += self.noise_array[self.datax.index(xelem)]
        ax.plot(x,y,"-",color = "crimson",alpha = 0.8)

    def calculate_peak(self,ROI:tuple)->object:
        x = self.datax[ROI[0]:ROI[1]+1]
        y = self.data_noiseless[ROI[0]:ROI[1]+1]
        return Gaussian(xdata=x, ydata=y)

    def calculate_peaks(self)->None:
        self.noise_array = self.remove_noise()
        for index,ROI in enumerate(self.ROI_limit):
            if len(self.peaks) != len(self.ROI_limit) :
                self.peaks.append(self.calculate_peak(ROI))
            else:
                self.peaks[index] = self.calculate_peak(ROI)

    def calibrate(self,echantillon:str,energie:list,plot = False)->None:
        centroid = [0]+[self.peaks[i].mu for i in range(len(self.peaks))]
        linFit =  Noise(centroid,[0]+energie)
        yfunction =linFit.get_noise()
        for index, peak in enumerate(self.peaks):
            peak.mucalib = yfunction(peak.mu)
            peak.sigma = yfunction(peak.sigma)
        self.calibrated = True
        if plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Canaux")
            ax.set_ylabel("Énergie [keV]")
            #ax.set_title("Courbe de calibration {}".format(echantillon))
            ax.text(0.2,0.8,"y = {}x + {}".format(round(linFit.param[0],4),round(linFit.param[1],4)),transform=ax.transAxes)
            ax.plot(self.datax,[yfunction(i) for i in range(len(self.datax))])
            ax.scatter(x=centroid, y=[0]+energie, marker="o",color = "red")
            for index, center in enumerate(centroid[1:]):
                ax.annotate(energie[index], (center, energie[index]))
            fig.savefig("Courbe de calibration {}.png".format(echantillon),bbox_inches = "tight")

class Noise():
    def __init__(self,xdata:list,ydata:list):
        self.xdata = xdata
        self.ydata = ydata
        self.param = self.linearFit(xdata,ydata)[0]

    @staticmethod
    def linearFit(xdata:list,ydata:list):
        return sp.curve_fit(lambda x,a,b:a*x+b,xdata=xdata,ydata=ydata)

    def get_noise(self)->callable:
        return lambda x :self.param[0]*x + self.param[1]


class Gaussian():
    def __init__(self,xdata:list,ydata:list):
        self.x = xdata
        self.datay = ydata
        param,pcov = sp.curve_fit(lambda x,mu,sigma,a: a*np.exp(-(x-mu)**2/(2*sigma**2)),xdata,ydata,p0=[np.mean(self.x),np.std(self.x),1])
        self.mu = param[0]
        self.sigma = abs(param[1])
        self.mucalib = param[0]
        self.sigmacalib = abs(param[1])
        self.a = param[2]
        self.y = [self.gaussian_function(x) for x in self.x]
    def gaussian_function(self,x)->callable:
        return self.a*np.exp(-(x-self.mu)**2/(2*self.sigma**2))


def Fait_toute_les_figures():
    for filename in os.listdir(os.fsencode("Data")):
        end = "AMDC" if filename.endswith(".mcs") else "maelstro"
        spectre = BuildSpectre(filename, end)
        fig, ax = spectre.show_Spectre(title="")
        spectre.add_ROIS_to_fig(ax)
        print(spectre.get_FWHMs())
        for i in range(len(spectre.ROI_limit)):
            spectre.plot_gaussian_over(ax, i)

        spectre.save_fig(filename)
def sum_les_shit(files:list):
    rois = []
    Data = np.zeros(2048,int)
    for file in files:
        with open("Data/{}".format(file), "r") as datafile:
            rawstr = datafile.read().split("\n")
            ROI_index = rawstr.index("<<ROI>>")
            data_index = rawstr.index("<<DATA>>")
            config_index = rawstr.index("<<DPP CONFIGURATION>>")
            data_list = rawstr[data_index + 1:config_index - 1]
            ROI_list = rawstr[ROI_index + 1:data_index]
            CONFIG_list = rawstr[config_index + 1:]
            Data += np.array(data_list,int)
            rois+=ROI_list
    with open("Data/SI-PIN.mca","w") as outfile :
        outfile.write("<<ROI>>\n")
        for i in range(len(rois)):
            outfile.write(rois[i]+"\n")
        outfile.write("<<DATA>>\n")
        for i in range(len(Data)):
            outfile.write(str(Data[i])+"\n")
        outfile.write("<<DPP CONFIGURATION>>")

if __name__ == "__main__":
    name = "Co + Cs + Na + Co.Spe"
    spectre1 = BuildSpectre(name,"maelstro")

    energy = [122.06,511,661.66,1173.228,1332.492]
    spectre1.calculate_peaks()
    spectre1.calibrate(energie=energy, echantillon=name[:2],plot=True)
    fig, ax = spectre1.show_Spectre("")
    spectre1.add_ROIS_to_fig(ax)
    spectre1.get_FWHMs(ax)
    for i in range(len(spectre1.ROI_limit)):
        spectre1.plot_gaussian_over(ax,i)
    plt.tight_layout()
    #spectre1.save_fig("Co + Cs + Na + Co.png")
    plt.show()