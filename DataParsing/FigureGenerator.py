import matplotlib.pyplot as plt
import  numpy as np
import scipy.optimize as sp
class BuildSpectre():
    def __init__(self,fileName,type):
        self.extradata = None
        if type == "maelstro" :
            self.data,ROIs  =  self.__importData_gamma(fileName)
        if type == "AMDC":
             self.data,ROIs,self.extradata = self.__importData_xray(fileName)
        else:
            assert AssertionError(" Le type est inconnue")
        self.ROI_limit = []
        self.peaks = []
        self.data_noiseless = self.data.copy()
        for coord in ROIs:
            coords = coord.split(" ")
            self.ROI_limit.append((int(coords[0]),int(coords[1])))
        self.noise_array = self.calculate_peaks()
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
            calibration_index = rawstr.index("<<CALIBRATION>>")
            config_index = rawstr.index("<<DPP CONFIGURATION>>")
            data_list =  rawstr[data_index+1:config_index-1]
            ROI_list = rawstr[ROI_index+1:data_index]
            CONFIG_list = rawstr[config_index+1:]
            CALIB_list = rawstr[calibration_index+1:ROI_index]
        return np.array(data_list,dtype="int"),ROI_list,CALIB_list,CONFIG_list

    def show_Spectre(self,title="placeHolder")->tuple:
        fig,ax = plt.subplots()
        data = self.data
        ax.bar(x = range(0,len(data)),height = data,width=1)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Nombre d'impulsion")
        ax.set_title("{}".format(title))
        plt.margins(x=0,y=0)
        return fig,ax

    def save_fig(self,figureName:str)->None:
        plt.savefig(figureName)

    def add_ROIS_to_fig(self,ax):
        ROI_mask = np.zeros(len(self.data), dtype=int)
        for coords in self.ROI_limit:
            ROI_mask[coords[0]:coords[1]+1] = self.data[coords[0]:coords[1]+1]
        ax.bar(x = range(0,len(self.data)),height=ROI_mask,color = "darkslateblue",width = 1)

    def calculate_centroid(self)->list:
        centroid = []
        for ROI in self.ROI_limit:
            roi_poid = self.data[ROI[0]:ROI[1]+1]
            centroid.append(round(sum(roi_poid*list(range(ROI[0],ROI[1]+1)))/sum(roi_poid),2))
        return centroid

    def remove_noise(self)->object:
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
        front = self.data[frontier[0]-3:frontier[0]+2]
        back = self.data[frontier[1]-1:frontier[1]+4]
        return np.concatenate((front,back),axis=None)

    def get_frontier_points_x(self,ROInumber:int)->list:
        frontier = self.ROI_limit[ROInumber]
        return list(range(frontier[0]-3,frontier[0]+2))+list(range(frontier[1]-1,frontier[1]+4))

    def add_ROI(self,leftlimit:int,rightlimit:int)->None:
        self.ROI_limit.append((leftlimit,rightlimit))

    def get_FWHMs(self)->list:
        FWHM = []
        for i in range(len(self.ROI_limit)):
            FWHM.append(self.get_FWHM(i))
        return FWHM

    def get_FWHM(self,index:int)->float:
        peak = self.peaks[index]
        return (peak.sigma)*np.sqrt(2*np.log(2))

    def plot_gaussian_over(self,ax,index:int):
        peak = self.peaks[index]
        x = peak.x
        y = peak.y
        for index,xelem in enumerate(x):
            y[index] += self.noise_array[xelem]
        ax.plot(x,y,"--",color = "red")

    def calculate_peak(self,ROI:tuple)->object:
        x = list(range(ROI[0], ROI[1] + 1))
        return Gaussian(xdata=x, ydata=[self.data_noiseless[i] for i in x])

    def calculate_peaks(self)->object:
        noise_array = self.remove_noise()
        for ROI in self.ROI_limit:
            self.peaks.append(self.calculate_peak(ROI))
        return noise_array
    def calibrate(self):
        pass
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
        param,pcov = sp.curve_fit(lambda x,mu,sigma,a: a*np.exp(-(x-mu)**2/(2*sigma**2)),xdata,ydata,p0=[np.mean(self.x),np.std(self.x),1])
        self.mu = param[0]
        self.sigma = abs(param[1])
        self.a = param[2]
        self.y = [self.gaussian_function(x) for x in self.x]
    def gaussian_function(self,x)->callable:
        return self.a*np.exp(-(x-self.mu)**2/(2*self.sigma**2))



if __name__ == "__main__":
    spectre = BuildSpectre("Co + Cs.Spe","maelstro")
    #spectre.remove_noise()
    fig,ax = spectre.show_Spectre()
    spectre.add_ROIS_to_fig(ax)
    print(spectre.get_FWHMs())
    spectre.plot_gaussian_over(ax,0)
    spectre.plot_gaussian_over(ax, 1)
    spectre.plot_gaussian_over(ax, 2)
    plt.show()
