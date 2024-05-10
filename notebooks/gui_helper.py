import pyvisa
import matplotlib.pyplot as plt
from ipywidgets import  widget_int, Dropdown, interactive_output
from chipwhisperer.common.api.ProjectFormat import Project
from IPython.display import display
from tqdm.notebook import tqdm as tqdm_notebook
import serial.tools.list_ports

VisaAddress = None
def set_visa_address(change):
    global VisaAddress
    if change['new'] == 'None':
        VisaAddress = None
    else:
        VisaAddress = change['new']

def get_visa_address():
    return VisaAddress

def get_inst_sel():
    inst_sel = Dropdown(options=pyvisa.ResourceManager().list_resources(), description='Select VISA Instrument:')
    inst_sel.options = list(inst_sel.options) + ['None']
    inst_sel.observe(set_visa_address, names='value')
    return inst_sel

ComPort = None
def set_comport(change):
    global ComPort
    if change['new'] == 'None':
        ComPort = None
    else:
        ComPort = change['new']

def get_comport_sel():
    comport_sel = Dropdown(options=[comport.device for comport in serial.tools.list_ports.comports()], description='Select COM Port:')
    comport_sel.options = list(comport_sel.options) + ['None']
    comport_sel.observe(set_comport, names='value')
    return comport_sel

def get_comport():
    return ComPort

class TqdmWidget(widget_int._BoundedInt):
    """
        A widget that wraps tqdm for use in Jupyter notebooks.

        Example:
        ```
        pbar = TqdmWidget(description='progress', max=100)
        pbar.display()
        for i in range(100):
            # Do something ...
            pbar.update(1)
        ```
    """
    def __init__(self, *args, **kwargs):
        desc = kwargs.pop('description', '')
        super().__init__(*args, **kwargs)
        total = self.max
        self.progress_bar = tqdm_notebook(total=total, display=False)
        self.progress_bar.set_description(desc)

    def update(self, value):
        self.progress_bar.update(value)
        self.value = self.progress_bar.n

    def display(self):
        display(self.progress_bar.container)


def get_waveform_pane(project : Project , tqdm : TqdmWidget, draw_interval=10, resample_rate=None):
    def draw(count):
        if draw_interval != 1 and count % draw_interval != 1:
            plt.show()
        else:
            if not resample_rate is None:
                wavelen = len(project.waves[-1])
                wave = project.waves[-1].resample(resample_rate * wavelen)
            else:
                wave = project.waves[-1]
            plt.plot(wave)
            plt.show()

    return interactive_output(draw, {'count': tqdm})

