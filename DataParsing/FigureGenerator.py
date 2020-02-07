import matplotlib.pyplot as plt
import  numpy as np
import scipy.optimize as sp
class BuildSpectre():
    def __init__(self,fileName,type):
        self.extradata = None
        if type == "maelstro" :
            self.data,self.ROIs  =  self.__importData_gamma(fileName)
        elif type == "AMDC":
             self.data,self.ROIs,self.extradata = self.__importData_xray(fileName)
        self.ROI_limit = []
        for coord in self.ROIs:
            coords = coord.split(" ")
            self.ROI_limit.append((int(coords[0]),int(coords[1])))

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
        return fig,ax

    def save_fig(self,figureName:str)->None:
        plt.savefig(figureName)

    def add_ROIS(self,ax):
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

    def remove_noise(self)->None:
        for index,ROI in enumerate(self.ROI_limit):
            x = self.get_frontier_points_x(index)
            y = self.get_frontier_points_y(index)
            noise_y = Noise(xdata= x, ydata=y).get_noise()
            for index,i in enumerate(x):
                self.data[i] -= noise_y[index]

    def remove_ROI(self,index:int)->None:
        self.ROI_limit.pop(index)

    def get_frontier_points_y(self,ROInumber:int)->list:
        frontier = self.ROI_limit[ROInumber]
        front = self.data[frontier[0]-3:frontier[0]+4]
        back = self.data[frontier[1]-3:frontier[1]+4]
        return np.concatenate((front,back),axis=None)

    def get_frontier_points_x(self,ROInumber:int)->list:
        frontier = self.ROI_limit[ROInumber]
        return list(range(frontier[0]-3,frontier[0]+4))+list(range(frontier[1]-3,frontier[1]+4))

    def add_ROI(self,leftlimit:int,rightlimit:int)->None:
        self.ROI_limit.append((leftlimit,rightlimit))

class Noise():
    def __init__(self,xdata:list,ydata:list):
        self.xdata = xdata
        self.ydata = ydata
        self.param = self.linearFit(xdata,ydata)[0]

    @staticmethod
    def linearFit(xdata:list,ydata:list):
        return sp.curve_fit(lambda x,a,b:a*x+b,xdata=xdata,ydata=ydata)

    def get_noise(self)->list:
        return [self.param[0]*x + self.param[1] for x in self.xdata]
if __name__ == "__main__":
    spectre = BuildSpectre("Co + Cs.Spe","maelstro")
    spectre.remove_noise()
    fig,ax = spectre.show_Spectre()
    spectre.add_ROIS(ax)
    plt.show()