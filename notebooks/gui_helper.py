from panel import widget
import pyvisa
from io import BytesIO
from matplotlib.figure import Figure
from ipywidgets import  Checkbox, IntSlider, Image, Text, widget_int, Dropdown, HBox, VBox, Button, BoundedFloatText, Label, IntText
from IPython.display import display

from tqdm.notebook import tqdm as tqdm_notebook
import serial.tools.list_ports
import chipwhisperer as cw
from cw_plugins.scopes import *
from cw_plugins.targets import *
from ipyfilechooser import FileChooser
from chipwhisperer.common.api import ProjectFormat as project
import os
from weakref import WeakKeyDictionary


# Keep GUI selections and successfully applied channels separate: editing a
# selector does not change the channel already configured on the instrument.
_scope_channels = WeakKeyDictionary()


def _register_channel_selector(scope, role, selector):
    state = _scope_channels.setdefault(scope, {'selected': {}, 'applied': {}})
    state['selected'][role] = selector.value
    selector.observe(lambda change: state['selected'].__setitem__(role, change['new']), names='value')
    return state


def _check_channel_conflict(state, role, channel):
    other = 'trace' if role == 'trigger' else 'trigger'
    if channel in (state['selected'].get(other), state['applied'].get(other)):
        raise ValueError('Trigger Channel and Trace Channel must be different channels.')

# ==============  VISA Oscilloscope Management ==============
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
    inst_sel = Dropdown(options=pyvisa.ResourceManager().list_resources(), description='VISA Instrument:', style={"description_width": "initial"}, layout={"width": "max-content"})
    inst_sel.options = list(inst_sel.options) + ['None']
    inst_sel.observe(set_visa_address, names='value')
    # select default None option
    if len(inst_sel.options) > 0:
        inst_sel.value = inst_sel.options[0]
        set_visa_address({'new': inst_sel.value})
    
    return inst_sel

def showScopeSelector():
    display(get_inst_sel())

# ================  Scope Sampling rate Configuration Panel ==============
def get_sampling_rate_panel(scope):
    rate_unit_dict = {
        "GSa/s": 1e9,
        "MSa/s": 1e6,
        "kSa/s": 1e3,
    }
    rate_input = BoundedFloatText(value=1.0, min=0.0, max=1000.0, step = 0.1, description='Sampling Rate:', style={"description_width": "initial"}, layout={"width": "max-content"})
    rate_unit = Dropdown(value="GSa/s", options=list(rate_unit_dict.keys()))

    apply_button = Button(description="apply", button_style='', layout={"width": "max-content"})

    def apply_sampling_rate_config():
        rate = float(rate_input.value)
        rate *= rate_unit_dict[rate_unit.value]
        try:
            scope.set_sampling_rate(rate)
        except Exception as e:
            apply_button.description = "Error"
            apply_button.button_style = 'danger'

        apply_button.disabled = True
        apply_button.description = "applied"
        apply_button.button_style = 'success'

    def button_unclicked():
        apply_button.disabled = False
        apply_button.description = "apply"
        apply_button.button_style = ''

    rate_input.observe(lambda _: button_unclicked(), names='value')
    rate_unit.observe(lambda _: button_unclicked(), names='value')

    apply_button.on_click(lambda _: apply_sampling_rate_config())

    return HBox([rate_input, rate_unit, apply_button])


def showSamplingRateConfig(scope):
    display(get_sampling_rate_panel(scope))

# ================  Trigger Configuration Panel ==============
def get_trigger_panel(scope):
    mode_dict = {
        "rising": TriggerMode.EDGE_RISE,
        "falling": TriggerMode.EDGE_FALL,
        "either": TriggerMode.EDGE_ANY,
    }
    mode_sel = Dropdown(value="rising", options=list(mode_dict.keys()), description='Mode:')

    ch_sel = Dropdown(value=1, options=[i + 1 for i in range(scope.get_num_channels())], description='Channel:')
    channels = _register_channel_selector(scope, 'trigger', ch_sel)

    scale_input = BoundedFloatText(value=1.0, min=0.0, max=1000.0, step = 0.1, description='Scale:')
    scale_unit = Dropdown(value="V", options=["V", "mV"])

    # offset input
    offset_input = BoundedFloatText(value=0.0, min=-1000.0, max=1000.0, step = 0.1, description='Offset:')
    offset_unit = Dropdown(value="V", options=["V", "mV"])

    level_input = BoundedFloatText(
        value=1.0, min=-1000.0, max=1000.0, step=0.1,
        description="Trigger Level:", style={"description_width": "initial"},
        disabled=True,
    )
    level_unit = Dropdown(
        value=None, options=[("Auto (offset + scale)", None), "V", "mV"],
    )

    apply_button = Button(description="apply", button_style='', layout={"width": "max-content"})
    apply_button.disabled = False

    msg = Label(value="")

    def apply_trigger_config():
        mode = mode_dict[mode_sel.value]
        channel = int(ch_sel.value)
        scale = float(scale_input.value)
        if scale_unit.value == "mV":
            scale /= 1000.0
        offset = float(offset_input.value)
        if offset_unit.value == "mV":
            offset /= 1000.0

        try:
            _check_channel_conflict(channels, 'trigger', channel)
            threshold = None
            if level_unit.value is not None:
                threshold = float(level_input.value)
                if level_unit.value == "mV":
                    threshold /= 1000.0
            scope.config_trigger_channel(
                channel, scale, offset, mode,
                threshold=threshold,
            )
        except Exception as e:
            msg.value = f"Error: {str(e)}"
            return
    
        channels['applied']['trigger'] = channel
        msg.value = ''
        apply_button.disabled = True
        apply_button.description = "applied"
        apply_button.button_style = 'success'

    def button_unclicked():
        apply_button.disabled = False
        apply_button.description = "apply"
        apply_button.button_style = ''
        msg.value = ""

    # any change in the inputs will trigger the apply function
    mode_sel.observe(lambda _: button_unclicked(), names='value')
    ch_sel.observe(lambda _: button_unclicked(), names='value')
    scale_input.observe(lambda _: button_unclicked(), names='value')
    scale_unit.observe(lambda _: button_unclicked(), names='value')
    offset_input.observe(lambda _: button_unclicked(), names='value')
    offset_unit.observe(lambda _: button_unclicked(), names='value')
    level_input.observe(lambda _: button_unclicked(), names='value')
    def level_unit_changed(_):
        level_input.disabled = level_unit.value is None
        button_unclicked()

    level_unit.observe(level_unit_changed, names='value')

    apply_button.on_click(lambda _: apply_trigger_config())

    return VBox([mode_sel, ch_sel, HBox([scale_input, scale_unit]), HBox([offset_input, offset_unit]), HBox([level_input, level_unit]), HBox([apply_button, msg])])

def showTriggerConfig(scope):
    display(get_trigger_panel(scope))

# ==============  Trace Configuration Panel ==============
def get_trace_panel(scope):
    time_scale_dict = {
        "s": 1,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }
    ch_sel = Dropdown(value=1, options=[i + 1 for i in range(scope.get_num_channels())], description='Channel:')
    channels = _register_channel_selector(scope, 'trace', ch_sel)

    scale_input = BoundedFloatText(value=1.0, min=0.0, max=100.0, step = 0.1, description='Scale:')
    scale_unit = Dropdown(value="mV", options=["V", "mV"])

    # offset input
    offset_input = BoundedFloatText(value=0.0, min=-1000.0, max=1000.0, step = 0.1, description='Offset:')
    offset_unit = Dropdown(value="V", options=["V", "mV"])

    period_input = BoundedFloatText(value=1, min=-1000, max=1000, step = 0.1, description='Period:')
    period_unit = Dropdown(value="us", options=time_scale_dict.keys())
    
    delay_input = BoundedFloatText(value=0.0, min=-1000.0, max=1000.0, step = 0.1, description='Delay:')
    delay_unit = Dropdown(value="us", options=time_scale_dict.keys())

    impedance_sel = Dropdown(
        options=[("Keep current", None), ("50 Ω", 50), ("1 MΩ", 1000000)],
        value=None, description="Input Impedance:",
        style={"description_width": "initial"}, layout={"width": "max-content"})

    apply_button = Button(description="apply", button_style='', layout={"width": "max-content"})
    apply_button.disabled = False

    msg = Label(value="")

    def apply_trace_config():
        channel = int(ch_sel.value)
        scale = float(scale_input.value)
        if scale_unit.value == "mV":
            scale /= 1000.0
        offset = float(offset_input.value)
        if offset_unit.value == "mV":
            offset /= 1000.0
        period = float(period_input.value)
        period *= time_scale_dict[period_unit.value]
        delay = float(delay_input.value)
        delay *= time_scale_dict[delay_unit.value]

        try:
            _check_channel_conflict(channels, 'trace', channel)
            kwargs = {}
            if impedance_sel.value is not None:
                kwargs["impedance"] = impedance_sel.value
            scope.config_trace_channel(channel, scale, offset, period, delay, **kwargs)
        except Exception as e:
            msg.value = f"Error: {str(e)}"
            return

        channels['applied']['trace'] = channel
        msg.value = ''
        apply_button.disabled = True
        apply_button.description = "applied"
        apply_button.button_style = 'success'

    def button_unclicked():
        apply_button.disabled = False
        apply_button.description = "apply"
        apply_button.button_style = ''
        msg.value = ""

    # any change in the inputs will trigger the apply function
    ch_sel.observe(lambda _: button_unclicked(), names='value')
    scale_input.observe(lambda _: button_unclicked(), names='value')
    scale_unit.observe(lambda _: button_unclicked(), names='value')
    offset_input.observe(lambda _: button_unclicked(), names='value')
    offset_unit.observe(lambda _: button_unclicked(), names='value')
    period_input.observe(lambda _: button_unclicked(), names='value')
    period_unit.observe(lambda _: button_unclicked(), names='value')
    delay_input.observe(lambda _: button_unclicked(), names='value')
    delay_unit.observe(lambda _: button_unclicked(), names='value')
    impedance_sel.observe(lambda _: button_unclicked(), names='value')

    apply_button.on_click(lambda _: apply_trace_config())

    return VBox([ch_sel, HBox([scale_input, scale_unit]), HBox([offset_input, offset_unit]), HBox([period_input, period_unit]), HBox([delay_input, delay_unit]), impedance_sel, HBox([apply_button, msg])])

def showTraceConfig(scope):
    display(get_trace_panel(scope))


# ============== Board Settings Panel ==============
class BoardSettingsPanel:
    STYLE = {"style": {"description_width": "initial"}, "layout": {"width": "max-content"}}
    def __init__(self):
        
        self.board_sel = Dropdown(value="SAKURA-X", options=["SAKURA-X", "CW305"], description='Select Board:', **self.STYLE)

        self.target_sel = Dropdown(value="AES RTL", options=["AES RTL", "AES HLS", "AES VexRiscV"], description='Select Target Type:', **self.STYLE)

        # sakura-x options
        self.sakurax_opt_label = Label(value="Options for SAKURA-X Board:")
        ports = [''] + [port.device for port in serial.tools.list_ports.comports()]
        self.data_port_sel = Dropdown(options=ports, value='', description='Data Port:', **self.STYLE)
        self.reset_port_sel = Dropdown(options=ports, value='', description='Reset Port:', **self.STYLE)
        self.auto_detect = Checkbox(value=False, description='Auto detect')
        self.sakurax_options = VBox([
            self.sakurax_opt_label, self.auto_detect,
            self.data_port_sel, self.reset_port_sel,
        ])

        def update_auto_detect(change):
            enabled = change['new']
            self.data_port_sel.disabled = enabled
            self.reset_port_sel.disabled = enabled

        self.auto_detect.observe(update_auto_detect, names='value')

        # rtl options
        self.rtl_opt_label = Label(value="Options for RTL Implementation:")
        self.rtl_impl_sel = Dropdown(value="aist", options=["aist", "google", "rsm"], description='Select RTL Core:', **self.STYLE)

        # soft options
        self.soft_opt_label = Label(value="Options for Software Implementation:")
        self.soft_opt_label.layout.display = 'none'
        self.soft_masking = Checkbox(value=False, description='Enable Masking')
        self.soft_masking.layout.display = 'none'

        # CW305 board options
        self.cw305_opt_label = Label(value="Options for CW305 Board:")
        self.cw305_opt_label.layout.display = 'none'
        self.bitfile_sel = FileChooser(title="Bitfile:", filter_pattern="*.bit", show_hidden=False)
        self.bitfile_sel.layout.display = 'none'
        self.hwhfile_sel = FileChooser(title="HWH file:", filter_pattern="*.hwh", show_hidden=False)
        self.hwhfile_sel.layout.display = 'none'


        def update_board_options(change):
            if change['new'] == "SAKURA-X":
                self.sakurax_options.layout.display = "block"
                self.bitfile_sel.layout.display = "none"
                self.hwhfile_sel.layout.display = "none"
                self.cw305_opt_label.layout.display = 'none'
            elif change['new'] == "CW305":
                self.sakurax_options.layout.display = "none"
                self.bitfile_sel.layout.display = "block"
                self.hwhfile_sel.layout.display = "block"
                self.cw305_opt_label.layout.display = 'block'
            else:
                self.sakurax_options.layout.display = "none"
                self.bitfile_sel.layout.display = "none"
                self.hwhfile_sel.layout.display = "none"
                self.cw305_opt_label.layout.display = 'none'

        def update_target_options(change):
            if change['new'] == "AES RTL":
                self.rtl_impl_sel.layout.display = 'block'
                self.rtl_opt_label.layout.display = 'block'
                self.soft_masking.layout.display = 'none'
                self.soft_opt_label.layout.display = 'none'
            elif change['new'] == "AES VexRiscV":
                self.rtl_impl_sel.layout.display = 'none'
                self.rtl_opt_label.layout.display = 'none'
                self.soft_masking.layout.display = 'block'
                self.soft_opt_label.layout.display = 'block'
            else:
                self.rtl_impl_sel.layout.display = 'none'
                self.rtl_opt_label.layout.display = 'none'
                self.soft_masking.layout.display = 'none'
                self.soft_opt_label.layout.display = 'none'

        self.board_sel.observe(lambda change: update_board_options(change), names='value')
        self.target_sel.observe(lambda change: update_target_options(change), names='value')

    def show(self):
        display(VBox([self.board_sel, self.target_sel, self.sakurax_options, self.cw305_opt_label, self.bitfile_sel, self.hwhfile_sel, self.rtl_opt_label, self.rtl_impl_sel, self.soft_opt_label, self.soft_masking]))

boardPanel = None
def showBoardSettingsPanel():
    global boardPanel
    if boardPanel is None:
        boardPanel = BoardSettingsPanel()
    boardPanel.show()

def connectBoard(scope):
    if boardPanel is None:
        raise Exception("Board settings panel is not initialized.")
    board_type = boardPanel.board_sel.value

    target = None

    if board_type == "SAKURA-X":
        ports = {}
        if not boardPanel.auto_detect.value:
            ports = dict(data_port=boardPanel.data_port_sel.value, reset_port=boardPanel.reset_port_sel.value)
            if not all(ports.values()):
                raise ValueError('Select both Data Port and Reset Port, or enable Auto detect.')
            if ports['data_port'] == ports['reset_port']:
                raise ValueError('Data Port and Reset Port must be different ports.')
        if boardPanel.target_sel.value == "AES RTL":
            rtl_impl = boardPanel.rtl_impl_sel.value
            target = cw.target(scope, SakuraXShellExampleAES128BitRTL, **ports, implementation=rtl_impl)
        elif boardPanel.target_sel.value == "AES HLS":
            target = cw.target(scope, SakuraXShellExampleAES128BitHLS, **ports)
        elif boardPanel.target_sel.value == "AES VexRiscV":
            target = cw.target(scope, SakuraXVexRISCVAESExample, **ports, masked = boardPanel.soft_masking.value)

    elif board_type == "CW305":
        kwargs = {}
        if boardPanel.bitfile_sel.selected is not None:
            kwargs['bs_file'] = boardPanel.bitfile_sel.selected
        if boardPanel.hwhfile_sel.selected is not None:
            kwargs['hwh_file'] = boardPanel.hwhfile_sel.selected
        
        if boardPanel.target_sel.value == "AES RTL":
            rtl_impl = boardPanel.rtl_impl_sel.value
            target = cw.target(scope, CW305ShellExampleAES128BitRTL, implementation=rtl_impl, **kwargs)
        elif boardPanel.target_sel.value == "AES HLS":
            target = cw.target(scope, CW305ShellExampleAES128BitHLS, **kwargs)
        elif boardPanel.target_sel.value == "AES VexRiscV":
            kwargs['masked'] = boardPanel.soft_masking.value
            target = cw.target(scope, CW305VexRISCVAESExample, **kwargs)

    if target is None:
        raise Exception("Failed to connect to the target. Please check the board settings and try again.")

    return target

class CapturePanel:
    def __init__(self, scope, target):
        self.scope = scope
        self.target = target
        # capture setting
        self.ktp = cw.ktp.Basic()
        self.key_label = Label(value="")
        self.key = self.ktp.next_key()
        self.show_key()
        self.key_gen_button = Button(description="Change Key", button_style='', layout={"width": "max-content"})
        self.key_gen_button.on_click(lambda _: self.change_key())

        self.trace_count_input = IntText(value=100, step = 100, description='Number of Traces:', style={"description_width": "initial"}, layout={"width": "max-content"})

        self.draw_interval = IntSlider(value = 5, min = 1, max = 100, step = 1, description = 'Draw Interval:', style={"description_width": "initial"}, layout={"width": "300px"})

        self.start_button = Button(description="Start Capture", button_style='', layout={"width": "max-content"})
        self.start_button.on_click(lambda _: self.capture())

        # progress bar
        self.progress_bar = tqdm_notebook(total=self.trace_count_input.value, display=False)
        
        def update_trace_count(change):
            count = change['new']
            if count < 1:
                self.trace_count_input.value = 1
                count = 1

            self.progress_bar.total = count
            self.progress_bar.n = 0
            for widget in self.progress_bar.container.children:
                if hasattr(widget, "max"):
                    widget.max = change['new']
        
            self.progress_bar.refresh()
    
        self.trace_count_input.observe(update_trace_count, names='value')

        # waveform plotting
        self.plot_output = Image(format="png")

        # project save directory
        self.save_dir_chooser = FileChooser(title="Select Save Directory:", show_hidden=False, select_default=True)
        self.save_dir_chooser.show_only_dirs = True
        self.save_dir_chooser.use_dir_icons = True

        self.project_name_input = Text(value="project", description='Project Name:', style={"description_width": "initial"}, layout={"width": "max-content"})


        self.overwrite_checkbox = Checkbox(value=False, description='Overwrite if exists', style={"description_width": "initial"}, layout={"width": "max-content"})


        self.save_button = Button(description="Save Project", button_style='', layout={"width": "max-content"})
        self.save_button.on_click(lambda _: self.save_project())

        self.save_msg = Label(value="")

        self.deactivate_save_widgets()
        
        
        self.project = project.Project()
        self._view = VBox([
            HBox([self.key_label, self.key_gen_button]), self.trace_count_input,
            self.draw_interval, self.start_button, self.progress_bar.container,
            self.plot_output, self.save_dir_chooser, self.project_name_input,
            HBox([self.overwrite_checkbox, self.save_button]), self.save_msg,
        ])
        self._display_handle = None

    def draw_waveform(self):
        # Render without registering a pyplot figure or publishing cell output.
        # Updating one image widget replaces the previous waveform in place.
        fig = Figure()
        ax = fig.subplots()
        ax.plot(self.project.waves[-1])
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png")
            self.plot_output.value = buffer.getvalue()

    def activate_save_widgets(self):
        self.save_dir_chooser.layout.display = 'block'
        self.overwrite_checkbox.layout.display = 'block'
        self.save_button.layout.display = 'block'
        self.save_msg.layout.display = 'block'
        self.save_button.disabled = False
        self.project_name_input.layout.display = 'block'

    def deactivate_save_widgets(self):
        self.save_dir_chooser.layout.display = 'none'
        self.overwrite_checkbox.layout.display = 'none'
        self.save_button.layout.display = 'none'
        self.save_msg.layout.display = 'none'
        self.save_button.disabled = True
        self.project_name_input.layout.display = 'none'

    def activate_config_widgets(self):
        self.trace_count_input.disabled = False
        self.draw_interval.disabled = False
        self.trace_count_input.disabled = False
        self.start_button.disabled = False
        self.key_gen_button.disabled = False

    def deactivate_config_widgets(self):
        self.trace_count_input.disabled = True
        self.draw_interval.disabled = True
        self.trace_count_input.disabled = True
        self.start_button.disabled = True
        self.key_gen_button.disabled = True

    def save_project(self):
        filename = self.save_dir_chooser.selected + "/" + self.project_name_input.value
        filename = project.ensure_cwp_extension(filename)
        
        if os.path.isfile(filename) and (self.overwrite_checkbox.value is False):
            self.save_msg.value = f"Error: File {filename} already exists. Enable overwrite to replace it."
        else:
    
            # If the user gives a relative path including ~, expand to the absolute path
            filename = os.path.abspath(os.path.expanduser(filename))

            self.project.setFilename(filename)
            self.project.save()
            self.save_msg.value = f"Project saved to {filename}"
    

    def change_key(self):
        self.ktp.fixed_key = False
        self.key = self.ktp.next_key()
        self.show_key()

    def show_key(self):
        s = self.key.hex()
        formatted = " ".join(s[i:i+2] for i in range(0, len(s), 2))
        self.key_label.value = f"Target Key: {formatted}"

    def show(self):
        if self._display_handle is None:
            self._display_handle = display(self._view, display_id=True)
        else:
            self._display_handle.update(self._view)

    def capture(self):
        # lock config widgets
        self.deactivate_config_widgets()

        # hide save panel and disable save button
        self.deactivate_save_widgets()

        num_traces = self.trace_count_input.value
        draw_interval = self.draw_interval.value

        # activate progress bar
        self.progress_bar.reset()

        self.project = project.Project()

        while len(self.project.traces) < num_traces:
            text = self.ktp.next_text()
            trace = cw.capture_trace(self.scope, self.target, text, self.key, as_int = True)
            if trace is None:
                continue
            print("trace length: ", len(trace))
            self.project.traces.append(trace)
            self.progress_bar.update(1)

            if len(self.project.traces) % draw_interval == 0:
                # draw waveform
                self.draw_waveform()

        # unlock config widgets
        self.activate_config_widgets()

        # show save panel and enable save button
        self.activate_save_widgets()

capturePanel = None
def showCapturePanel(scope, target):
    global capturePanel
    if capturePanel is None:
        capturePanel = CapturePanel(scope, target)
    capturePanel.show()        


# ============== ChipWhisperer Lite / Husky Scope Settings ==============
class CWScopeSettingsPanel:
    """Configure an already connected cw.scope(); no settings change until apply.

    Sampling rate is generated from CLKGEN and the ADC multiplier. Changing it
    also changes the clock on HS2 when that output is enabled.
    """
    STYLE = {"style": {"description_width": "initial"},
             "layout": {"width": "max-content"}}

    def __init__(self, scope):
        self.scope = scope
        name = scope.get_name()
        if name not in ("ChipWhisperer Lite", "ChipWhisperer Husky"):
            raise ValueError("Only ChipWhisperer Lite and Husky are supported.")
        self.is_husky = name == "ChipWhisperer Husky"
        clock, adc = scope.clock, scope.adc
        rate = float(clock.adc_freq) / int(adc.decimate)
        self.rate_input = BoundedFloatText(
            value=max(0.001, rate / 1e6), min=0.001,
            max=200 if self.is_husky else 105, step=0.1,
            description='Sampling Rate (MSa/s):', **self.STYLE)
        self.time_units = {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
        initial_rate = rate if rate > 0 else self.rate_input.value * 1e6

        def time_field(value, description, minimum=0):
            field = BoundedFloatText(value=value / initial_rate * 1e6,
                                     min=minimum, max=1e12, step=0.1,
                                     description=description, **self.STYLE)
            unit = Dropdown(value='us', options=list(self.time_units),
                            layout={'width': '65px', 'min_width': '65px'})
            return field, unit

        self.length_input, self.length_unit = time_field(adc.samples, 'Trace Length:')
        self.length_input.value = 10.0
        initial_delay = -adc.presamples if adc.presamples else adc.offset
        self.delay_input, self.delay_unit = time_field(initial_delay, 'Delay:', minimum=-1e12)
        self.gain_input = BoundedFloatText(value=float(scope.gain.db), min=-6.5, max=56, step=0.5,
                                          description='Gain (dB):', **self.STYLE)
        self.trigger_input = Text(value=scope.trigger.triggers, description='Trigger Pins:', **self.STYLE)
        self.mode_input = Dropdown(value='rising_edge', options=['rising_edge', 'falling_edge', 'high', 'low'],
                                   description='Trigger Mode:', **self.STYLE)
        self.timeout_input = BoundedFloatText(value=float(adc.timeout), min=0.001, max=1e6, step=0.1,
                                             description='Timeout (s):', **self.STYLE)
        self.apply_button = Button(description='apply', layout={"width": "max-content"})
        self.msg = Label(value='')
        self.inputs = [self.rate_input, self.length_input, self.length_unit,
                       self.delay_input,
                       self.delay_unit, self.gain_input,
                       self.trigger_input, self.mode_input, self.timeout_input]
        for item in self.inputs:
            item.observe(self._changed, names='value')
        self.apply_button.on_click(self.apply)
        self.panel = VBox([Label(value=f'Detected board: {name}'),
                           self.rate_input,
                           HBox([self.length_input, self.length_unit,
                                 self.delay_input, self.delay_unit]),
                           self.gain_input, self.trigger_input, self.mode_input,
                           self.timeout_input, HBox([self.apply_button, self.msg])])

    def _changed(self, change):
        self.apply_button.disabled = False
        self.apply_button.description = 'apply'
        self.apply_button.button_style = ''
        self.msg.value = ''

    def apply(self, button=None):
        scope = self.scope
        try:
            length = self.length_input.value * self.time_units[self.length_unit.value]
            signed_delay = self.delay_input.value * self.time_units[self.delay_unit.value]
            pretrigger = max(0.0, -signed_delay)
            delay = max(0.0, signed_delay)
            durations = [length, pretrigger, delay]
            if length <= 0:
                raise ValueError('Trace Length must be positive.')
            if not 0 <= pretrigger < length:
                raise ValueError('Pretrigger must be nonnegative and smaller than Trace Length.')
            if not self.trigger_input.value.strip():
                raise ValueError('Specify Trigger Pins, for example tio4.')
            # Husky uses the same x1 ADC clock as the working SAKURA test.
            # Lite retains its current supported x1/x4 source.
            multiplier = (1 if self.is_husky else
                          (4 if scope.clock.adc_src.endswith('x4') else 1))
            if multiplier < 1:
                raise ValueError('ADC clock is disabled; initialize the scope clock first.')
            frequency = self.rate_input.value * 1e6 * int(scope.adc.decimate) / multiplier
            if frequency < 3.2e6:
                raise ValueError('Requested sampling rate is too low for the current ADC clock configuration.')
            scope.clock.clkgen_src = 'system'
            if self.is_husky:
                scope.clock.adc_mul = 1
            else:
                scope.clock.adc_src = f'clkgen_x{multiplier}'
            scope.clock.clkgen_freq = frequency
            scope.clock.reset_dcms()
            if not scope.clock.adc_locked:
                raise RuntimeError('ADC clock is not locked. Wait briefly and apply again.')
            rate = float(scope.clock.adc_freq) / int(scope.adc.decimate)
            if rate <= 0:
                raise RuntimeError('ADC sampling rate is unavailable. Apply again after clock stabilization.')
            # Use the actual rate, since synthesized clocks can differ from the
            # requested rate. Round to the nearest whole sample (half up).
            samples, presamples, offset = [int(t * rate + 0.5) for t in durations]
            maximum = getattr(scope.adc, 'max_samples', 131124 if self.is_husky else 24400)
            if not 1 <= samples <= maximum:
                raise ValueError(f'Trace Length must resolve to 1–{maximum} samples at {rate / 1e6:.6g} MSa/s.')
            if not 0 <= presamples < samples:
                raise ValueError('Pretrigger must resolve to fewer samples than Trace Length.')
            scope.adc.presamples = 0
            scope.adc.samples = samples
            scope.adc.presamples = presamples
            scope.adc.offset = offset
            scope.adc.timeout = self.timeout_input.value
            scope.gain.db = self.gain_input.value
            if self.is_husky:
                scope.trigger.module = 'basic'
            # Leave trigger pins as inputs, as in the CW acquisition notebook.
            for pin in ('tio1', 'tio2', 'tio3', 'tio4'):
                if pin in self.trigger_input.value.lower().split():
                    setattr(scope.io, pin, 'high_z')
            scope.trigger.triggers = self.trigger_input.value
            scope.adc.basic_mode = self.mode_input.value
            rate = float(scope.clock.adc_freq) / int(scope.adc.decimate)
            self.msg.value = (f'Applied: {rate / 1e6:.6g} MSa/s, {scope.adc.samples} samples, '
                              f'length {scope.adc.samples / rate * 1e6:.6g} us, '
                              f'delay {(scope.adc.offset - scope.adc.presamples) / rate * 1e6:.6g} us')
        except Exception as error:
            self.apply_button.disabled = False
            self.apply_button.description = 'apply'
            self.apply_button.button_style = 'danger'
            self.msg.value = f'Error: {error} (some settings may already have changed; correct and apply again.)'
            return
        self.apply_button.disabled = True
        self.apply_button.description = 'applied'
        self.apply_button.button_style = 'success'

    def show(self):
        display(self.panel)


def get_cw_scope_panel(scope):
    """Return a Lite/Husky settings widget for an existing cw.scope()."""
    return CWScopeSettingsPanel(scope).panel


def showCWScopeConfig(scope):
    """Display Lite/Husky setup: scope = cw.scope(); showCWScopeConfig(scope)."""
    panel = CWScopeSettingsPanel(scope)
    panel.show()
