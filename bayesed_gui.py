import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import subprocess
import threading
import queue
import sys
from PIL import Image, ImageDraw, ImageTk, ImageFont
import pyperclip
import os
import psutil
from datetime import datetime
from bayesed import *
import webbrowser

class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x = y = 0
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background='#FFFFDD', relief='solid', borderwidth=1,
                         font=("Arial", "12", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class BayeSEDGUI:
    def __init__(self, master):
        self.master = master
        master.title("BayeSED3 GUI")
        master.geometry("1400x800")
        
        # Define a standard font
        self.standard_font = ('Helvetica', 14)
        
        # Apply the standard font to all ttk widgets
        style = ttk.Style()
        style.configure('.', font=self.standard_font)
        
        self.galaxy_count = -1  # Start from -1, so the first instance will be 0
        
        # Initialize instances lists
        self.galaxy_instances = []
        self.agn_instances = []
        self.cosmology_params = {}
        self.igm_model = tk.StringVar()
        self.redshift_widgets = []
        self.redshift_params = {}
        
        # Initialize BooleanVar for checkboxes
        self.use_cosmology = tk.BooleanVar(value=False)
        self.use_igm = tk.BooleanVar(value=False)
        self.use_redshift = tk.BooleanVar(value=False)
        
        # Initialize BooleanVar for Advanced Settings checkboxes
        self.use_multinest = tk.BooleanVar(value=False)
        self.use_nnlm = tk.BooleanVar(value=False)
        self.use_ndumper = tk.BooleanVar(value=False)
        self.use_gsl = tk.BooleanVar(value=False)
        self.use_misc = tk.BooleanVar(value=False)
        self.use_sfr = tk.BooleanVar(value=False)
        self.use_snr = tk.BooleanVar(value=False)
        self.use_build_sedlib = tk.BooleanVar(value=False)
        self.no_photometry_fit = tk.BooleanVar(value=False)
        self.no_spectra_fit = tk.BooleanVar(value=False)
        self.unweighted_samples = tk.BooleanVar(value=False)
        self.priors_only = tk.BooleanVar(value=False)
        self.use_output_sfh = tk.BooleanVar(value=False)
        self.use_sys_err = tk.BooleanVar(value=False)
        self.sys_err_widgets = []
        
        # Initialize other necessary variables
        self.redshift_widgets = []
        
        # Create and set the icon
        self.create_icon()
        
        self.create_widgets()

        self.output_queue = queue.Queue()
        self.process = None

        # Add the stop_output_thread attribute
        self.stop_output_thread = threading.Event()

    def create_icon(self):
        try:
            # Load the provided icon
            image = Image.open("BayeSED3.jpg")
            
            # Resize the image to 128x128 for a higher resolution icon
            image = image.resize((128, 128), Image.LANCZOS)
            
            # Convert the image to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Set the window icon
            self.master.iconphoto(False, photo)
        except Exception as e:
            print(f"Error creating icon: {str(e)}")
            # If there's an error, we'll just skip setting the icon
            pass

    def create_widgets(self):
        # Create the About button first
        self.create_about_button()

        # 创建一个主框架来容纳所有元素
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建标签页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_basic_tab()
        self.create_galaxy_tab()
        self.create_AGN_tab()
        self.create_cosmology_tab()
        self.create_advanced_tab()

    def create_basic_tab(self):
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="Basic Settings")

        # Create left and right frames
        left_frame = ttk.Frame(basic_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        right_frame = ttk.Frame(basic_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Input settings (left frame)
        input_frame = ttk.LabelFrame(left_frame, text="Input Settings")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Apply the standard font to all widgets
        for widget in [ttk.Label, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Combobox]:
            style = ttk.Style()
            style.configure(f'{widget.__name__}.TLabel', font=self.standard_font)
            style.configure(f'{widget.__name__}.TEntry', font=self.standard_font)
            style.configure(f'{widget.__name__}.TButton', font=self.standard_font)
            style.configure(f'{widget.__name__}.TCheckbutton', font=self.standard_font)
            style.configure(f'{widget.__name__}.TCombobox', font=self.standard_font)

        # Input File
        ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_file = ttk.Entry(input_frame, width=40)
        self.input_file.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=2)
        CreateToolTip(self.input_file, "Input file containing observed photometric and/or spectroscopic SEDs")

        # Input Type
        ttk.Label(input_frame, text="Input Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_type = ttk.Combobox(input_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"], width=15)
        self.input_type.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.input_type.set("0 (flux in uJy)")
        CreateToolTip(self.input_type, "0: flux in uJy, 1: AB magnitude")

        # Output Directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.outdir = ttk.Entry(input_frame, width=40)
        self.outdir.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        self.outdir.insert(0, "result")
        ttk.Button(input_frame, text="Browse", command=self.browse_outdir).grid(row=2, column=2, padx=5, pady=2)
        CreateToolTip(self.outdir, "Output directory for all results")

        # Verbosity
        ttk.Label(input_frame, text="Verbosity:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.verbose = ttk.Combobox(input_frame, values=["0", "1", "2", "3"], width=5)
        self.verbose.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.verbose.set("2")
        CreateToolTip(self.verbose, "Verbose level (0-3)")

        # Filters
        ttk.Label(input_frame, text="Filters definition:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters = ttk.Entry(input_frame, width=40)
        self.filters.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_filters).grid(row=4, column=2, padx=5, pady=2)
        CreateToolTip(self.filters, "File containing the definition of filters")

        # Filters Selected
        ttk.Label(input_frame, text="Filters selection:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters_selected = ttk.Entry(input_frame, width=40)
        self.filters_selected.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_filters_selected).grid(row=5, column=2, padx=5, pady=2)
        CreateToolTip(self.filters_selected, "File containing all used filters in the observation and select those needed")

        # Priors Only
        ttk.Checkbutton(input_frame, text="Priors Only", variable=self.priors_only).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Test priors by setting the loglike for observational data to be zero")

        # No photometry fit
        ttk.Checkbutton(input_frame, text="No photometry fit", variable=self.no_photometry_fit).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Do not fit photometric data even if it is presented")

        # No spectra fit
        ttk.Checkbutton(input_frame, text="No spectra fit", variable=self.no_spectra_fit).grid(row=8, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Do not fit spectra data even if it is presented")

        # SNR Settings
        snr_frame = ttk.Frame(input_frame)
        snr_frame.grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(snr_frame, variable=self.use_snr, 
                        command=lambda: self.toggle_widgets([self.snrmin1, self.snrmin2], self.use_snr.get()),
                        text="SNR Settings").pack(side=tk.LEFT, padx=5)
        CreateToolTip(snr_frame.winfo_children()[-1], "Enable/disable SNR settings")

        snr_content = ttk.Frame(snr_frame)
        snr_content.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(snr_content, text="SNRmin1 (phot,spec):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.snrmin1 = ttk.Entry(snr_content, width=10)
        self.snrmin1.insert(0, "0,0")
        self.snrmin1.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(self.snrmin1, "The minimal SNR of data (phot,spec) to be used for determining scaling")
        
        ttk.Label(snr_content, text="SNRmin2 (phot,spec):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.snrmin2 = ttk.Entry(snr_content, width=10)
        self.snrmin2.insert(0, "0,0")
        self.snrmin2.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(self.snrmin2, "The minimal SNR of data (phot,spec) to be used for likelihood evaluation")

        # Initialize the SNR widgets to be disabled and grey
        self.toggle_widgets([self.snrmin1, self.snrmin2], False)

        # Systematic Error
        sys_err_frame = ttk.Frame(input_frame)
        sys_err_frame.grid(row=10, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(sys_err_frame, variable=self.use_sys_err, 
                        command=lambda: self.toggle_widgets(self.sys_err_widgets, self.use_sys_err.get()),
                        text="Systematic Error").pack(side=tk.LEFT)
        CreateToolTip(sys_err_frame.winfo_children()[-1], "Set priors for systematic errors of model and observation")

        sys_err_content = ttk.Frame(sys_err_frame)
        sys_err_content.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.sys_err_widgets = []
        for i, label in enumerate(["Mod", "Obs"]):
            ttk.Label(sys_err_content, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(5, 2))
            widgets = []
            for j, param in enumerate(["iprior_type", "is_age", "min", "max", "nbin"]):
                widget = ttk.Entry(sys_err_content, width=5)
                widget.grid(row=i, column=j+1, padx=2)
                widgets.append(widget)
                if j == 0:
                    widget.insert(0, "1")
                elif j == 1:
                    widget.insert(0, "0")
                elif j in [2, 3]:
                    widget.insert(0, "0")
                else:
                    widget.insert(0, "40")
            self.sys_err_widgets.extend(widgets)

        # Add tooltips for each parameter
        param_tooltips = [
            "Prior type (0-7)",
            "Age-dependent flag (0 or 1)",
            "Minimum systematic error",
            "Maximum systematic error",
            "Number of bins"
        ]

        for i, widget in enumerate(self.sys_err_widgets):
            CreateToolTip(widget, param_tooltips[i % 5])

        # Initialize the Systematic Error widgets to be disabled and grey
        self.toggle_widgets(self.sys_err_widgets, False)

        # Configure column weights for input_frame
        input_frame.columnconfigure(1, weight=1)

        # Save and Output options (right frame)
        save_output_frame = ttk.Frame(right_frame)
        save_output_frame.pack(fill=tk.BOTH, expand=True)

        # Save Options
        save_frame = ttk.LabelFrame(save_output_frame, text="Save Options")
        save_frame.pack(fill=tk.X, padx=5, pady=5)

        # Save Best Fit
        self.save_bestfit = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Best Fit", variable=self.save_bestfit).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save the best fitting result")

        self.save_bestfit_type = ttk.Combobox(save_frame, values=["0 (fits)", "1 (hdf5)", "2 (fits and hdf5)"], width=15, state="disabled")
        self.save_bestfit_type.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.save_bestfit_type.set("0 (fits)")
        CreateToolTip(self.save_bestfit_type, "Type of best fit output")

        # Enable/disable the combobox based on the checkbox
        self.save_bestfit.trace("w", lambda *args: self.save_bestfit_type.config(state="readonly" if self.save_bestfit.get() else "disabled"))

        # Save Parameter Posterior Sample
        self.save_sample_par = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Parameter Posterior Sample", variable=self.save_sample_par).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save the posterior sample of parameters")

        # Save Observable Posterior Sample
        self.save_sample_obs = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Observable Posterior Sample", variable=self.save_sample_obs).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save posteriori sample of observables")

        # Save Posterior SFH
        save_pos_sfh_frame = ttk.Frame(save_frame)
        save_pos_sfh_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.save_pos_sfh = tk.BooleanVar()
        ttk.Checkbutton(save_pos_sfh_frame, text="Save Posterior SFH", variable=self.save_pos_sfh).pack(side=tk.LEFT)
        CreateToolTip(save_pos_sfh_frame.winfo_children()[-1], "Save the posterior distribution of SFH from t=0 to t=tage")
        
        ttk.Label(save_pos_sfh_frame, text="Ngrid:").pack(side=tk.LEFT, padx=(5, 2))
        self.save_pos_sfh_ngrid = ttk.Entry(save_pos_sfh_frame, width=5)
        self.save_pos_sfh_ngrid.pack(side=tk.LEFT)
        self.save_pos_sfh_ngrid.insert(0, "100")
        CreateToolTip(self.save_pos_sfh_ngrid, "Number of grid points for SFH")
        
        ttk.Label(save_pos_sfh_frame, text="ilog:").pack(side=tk.LEFT, padx=(5, 2))
        self.save_pos_sfh_ilog = ttk.Combobox(save_pos_sfh_frame, values=["0", "1"], width=3)
        self.save_pos_sfh_ilog.pack(side=tk.LEFT)
        self.save_pos_sfh_ilog.set("1")
        CreateToolTip(self.save_pos_sfh_ilog, "0 for linear scale, 1 for log scale")

        # Save Posterior Spectra
        self.save_pos_spec = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Posterior Spectra", variable=self.save_pos_spec).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save the posterior distribution of model spectra (Warning: requires memory of size nSamples*Nwavelengths!)")

        # Save Sample Spectra
        self.save_sample_spec = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Sample Spectra", variable=self.save_sample_spec).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save the posterior sample of model spectra")

        # Save Summary
        self.save_summary = tk.BooleanVar()
        ttk.Checkbutton(save_frame, text="Save Summary", variable=self.save_summary).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(save_frame.winfo_children()[-1], "Save the summary file")

        # Output Options
        output_frame = ttk.LabelFrame(save_output_frame, text="Output Options")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        # Output mock photometry
        self.output_mock_photometry = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Output Mock Photometry", variable=self.output_mock_photometry).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Output mock photometry with best fit")

        self.output_mock_photometry_type = ttk.Combobox(output_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"], width=15, state="disabled")
        self.output_mock_photometry_type.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.output_mock_photometry_type.set("0 (flux in uJy)")
        CreateToolTip(self.output_mock_photometry_type, "Type of mock photometry output (0: flux in uJy, 1: AB magnitude)")

        # Enable/disable the combobox based on the checkbox
        self.output_mock_photometry.trace("w", lambda *args: self.output_mock_photometry_type.config(state="readonly" if self.output_mock_photometry.get() else "disabled"))

        # Output mock spectra
        self.output_mock_spectra = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Output Mock Spectra", variable=self.output_mock_spectra).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Output mock spectra (in uJy) with best fit")

        # Output model absolute magnitude
        self.output_model_absolute_magnitude = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Output Model Absolute Magnitude", variable=self.output_model_absolute_magnitude).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Output model absolute magnitude of best fit")

        # Output posterior observables
        self.output_pos_obs = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Output Posterior Observables", variable=self.output_pos_obs).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Output posterior estimation of observables")

        # Build SED Library
        build_sed_frame = ttk.Frame(output_frame)
        build_sed_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(build_sed_frame, variable=self.use_build_sedlib, 
                        command=lambda: self.toggle_widgets([self.build_sedlib], self.use_build_sedlib.get()),
                        text="Build SED Library").pack(side=tk.LEFT)
        CreateToolTip(build_sed_frame.winfo_children()[-1], "Build a SED library using the employed models")

        self.build_sedlib = ttk.Combobox(build_sed_frame, values=["0 (Rest)", "1 (Observed)"], width=15, state="disabled")
        self.build_sedlib.set("0 (Rest)")
        self.build_sedlib.pack(side=tk.LEFT, padx=5)
        CreateToolTip(self.build_sedlib, "0: Rest frame, 1: Observed frame")

        # Use unweighted samples
        ttk.Checkbutton(output_frame, text="Use unweighted samples", variable=self.unweighted_samples).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Use unweighted posterior samples")

        # SFR Settings
        sfr_frame = ttk.Frame(output_frame)
        sfr_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(sfr_frame, variable=self.use_sfr, 
                        command=lambda: self.toggle_widgets([self.sfr_myr_entry], self.use_sfr.get()),
                        text="Output SFR over").pack(side=tk.LEFT)
        CreateToolTip(sfr_frame.winfo_children()[-1], "Compute average SFR over the past given Myrs")

        ttk.Label(sfr_frame, text="Myr values:").pack(side=tk.LEFT, padx=(5, 2))
        self.sfr_myr_entry = ttk.Entry(sfr_frame, width=15)
        self.sfr_myr_entry.pack(side=tk.LEFT)
        self.sfr_myr_entry.insert(0, "10,100")
        CreateToolTip(self.sfr_myr_entry, "Comma-separated Myr values for SFR computation (e.g., 10,100 or 10,100,1000)")

        # Initialize the SFR widgets to be disabled and grey
        self.toggle_widgets([self.sfr_myr_entry], False)

        # Output SFH
        sfh_frame = ttk.Frame(output_frame)
        sfh_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(sfh_frame, variable=self.use_output_sfh, 
                        command=lambda: self.toggle_widgets([self.output_sfh_ntimes, self.output_sfh_ilog], self.use_output_sfh.get()),
                        text="Output SFH").pack(side=tk.LEFT)
        CreateToolTip(sfh_frame.winfo_children()[-1], "Output the SFH over the past tage year")

        ttk.Label(sfh_frame, text="ntimes:").pack(side=tk.LEFT, padx=(5, 2))
        self.output_sfh_ntimes = ttk.Entry(sfh_frame, width=5)
        self.output_sfh_ntimes.pack(side=tk.LEFT)
        self.output_sfh_ntimes.insert(0, "10")
        CreateToolTip(self.output_sfh_ntimes, "Number of time points")

        ttk.Label(sfh_frame, text="ilog:").pack(side=tk.LEFT, padx=(5, 2))
        self.output_sfh_ilog = ttk.Combobox(sfh_frame, values=["0", "1"], width=3)
        self.output_sfh_ilog.pack(side=tk.LEFT)
        self.output_sfh_ilog.set("0")
        CreateToolTip(self.output_sfh_ilog, "0 for linear scale, 1 for log scale")

        # Suffix
        ttk.Label(output_frame, text="Suffix:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=2)
        self.suffix = ttk.Entry(output_frame, width=30)
        self.suffix.grid(row=8, column=1, sticky=tk.EW, padx=5, pady=2)
        CreateToolTip(self.suffix, "Add suffix to the name of output file")

        # Configure column weights for output_frame
        output_frame.columnconfigure(1, weight=1)

        # Configure grid weights for basic_frame
        basic_frame.grid_columnconfigure(0, weight=1)
        basic_frame.grid_columnconfigure(1, weight=1)

        # 创建底部框架用于运行按钮、导入/导出设置和输出
        bottom_frame = ttk.Frame(basic_frame)
        bottom_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # 控制按钮框架
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # MPI processes input
        ttk.Label(control_frame, text="MPI processes:").pack(side=tk.LEFT, padx=(0, 5))
        self.mpi_processes = ttk.Entry(control_frame, width=5)
        self.mpi_processes.pack(side=tk.LEFT, padx=(0, 5))
        CreateToolTip(self.mpi_processes, "Number of MPI processes to use (optional, leave empty to use all cores)")

        # Ntest input (moved here)
        ttk.Label(control_frame, text="Ntest:").pack(side=tk.LEFT, padx=(5, 5))
        self.ntest = ttk.Entry(control_frame, width=5)
        self.ntest.pack(side=tk.LEFT, padx=(0, 5))
        CreateToolTip(self.ntest, "Number of objects for test run (leave empty to process all objects)")

        # Run button
        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_bayesed)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Add Save Script button
        save_script_button = ttk.Button(control_frame, text="Save Script", command=self.save_script)
        save_script_button.pack(side=tk.LEFT, padx=5)

        # Plot button and FITS file selection
        plot_frame = ttk.Frame(control_frame)
        plot_frame.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)

        ttk.Button(plot_frame, text="Plot", command=self.plot_bestfit).pack(side=tk.LEFT)
        
        self.fits_file = ttk.Entry(plot_frame, width=40)  # Increased width from 20 to 40
        self.fits_file.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        CreateToolTip(self.fits_file, "Path to the FITS file to plot")

        ttk.Button(plot_frame, text="Browse", command=self.browse_fits_file).pack(side=tk.LEFT, padx=(5, 0))

        # Output frame
        output_frame = ttk.LabelFrame(bottom_frame, text="Output")
        output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Clear button
        clear_button = ttk.Button(output_frame, text="Clear", command=self.clear_output)
        clear_button.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure the output_text widget to allow copying but prevent editing
        self.output_text.config(state=tk.NORMAL)
        self.output_text.bind("<Button-1>", lambda e: self.output_text.focus_set())
        self.output_text.bind("<Control-c>", self.copy_selection)
        self.output_text.bind("<Control-x>", self.copy_selection)
        self.output_text.bind("<Key>", lambda e: "break")

        # Configure grid weights
        basic_frame.grid_columnconfigure(0, weight=1)
        basic_frame.grid_columnconfigure(1, weight=1)
        basic_frame.grid_rowconfigure(0, weight=1)
        basic_frame.grid_rowconfigure(3, weight=1)

    def create_galaxy_tab(self):
        self.galaxy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.galaxy_frame, text="Galaxy")

        # Add button at the top
        ttk.Button(self.galaxy_frame, text="Add", command=self.add_galaxy_instance).pack(pady=5, anchor=tk.NW)

        # Container for galaxy instances
        self.galaxy_instances_frame = ttk.Frame(self.galaxy_frame)
        self.galaxy_instances_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.galaxy_instances = []
        self.add_galaxy_instance()

    def add_galaxy_instance(self):
        # Find the maximum ID and igroup among existing galaxy and AGN instances
        max_id = max_igroup = -1
        for instance in self.galaxy_instances:
            max_id = max(max_id, int(instance['ssp'][1].get()))  # id is the second element in ssp list
            max_id = max(max_id, int(instance['dem'][1].get()))  # Consider DEM ID as well
            max_igroup = max(max_igroup, int(instance['ssp'][0].get()))  # igroup is the first element in ssp list
        for instance in self.agn_instances:
            max_id = max(max_id, int(instance['agn_id'].get()))
            max_igroup = max(max_igroup, int(instance['agn_igroup'].get()))
        
        new_id = max_id + 2  # Increment by 2 to leave room for DEM ID
        new_dem_id = new_id + 1  # DEM ID is always one more than the main ID
        new_igroup = max_igroup + 1  # Increment igroup by 1
        
        instance_frame = ttk.LabelFrame(self.galaxy_instances_frame, text=f"CSP {len(self.galaxy_instances)}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        def update_ids(event):
            new_id = ssp_id_widget.get()
            sfh_id_widget.config(state='normal')
            sfh_id_widget.delete(0, tk.END)
            sfh_id_widget.insert(0, new_id)
            sfh_id_widget.config(state='readonly')

            dal_id_widget.config(state='normal')
            dal_id_widget.delete(0, tk.END)
            dal_id_widget.insert(0, new_id)
            dal_id_widget.config(state='readonly')

            dem_id_widget.config(state='normal')
            dem_id_widget.delete(0, tk.END)
            dem_id_widget.insert(0, str(int(new_id) + 1))  # Increment the ID by 1 for DEM
            dem_id_widget.config(state='readonly')

            kin_widgets['id'].config(state='normal')
            kin_widgets['id'].delete(0, tk.END)
            kin_widgets['id'].insert(0, new_id)
            kin_widgets['id'].config(state='readonly')

        # SSP settings
        ssp_frame = ttk.Frame(instance_frame)
        ssp_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ssp_frame, text="SSP:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        ssp_params = [
            ("igroup", str(new_igroup), 5),
            ("id", str(new_id), 5),
            ("name", "bc2003_hr_stelib_chab_neb_2000r", 30),
            ("iscalable", "1", 5),
            ("k", "1", 5),
            ("f_run", "1", 5),
            ("Nstep", "1", 5),
            ("i0", "0", 5),
            ("i1", "0", 5),
            ("i2", "0", 5),
            ("i3", "0", 5)
        ]
        
        ssp_widgets = []
        for i, (param, default, width) in enumerate(ssp_params):
            ttk.Label(ssp_frame, text=f"{param}:").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
            if param in ['iscalable']:
                widget = ttk.Combobox(ssp_frame, values=["0", "1"], width=width)
                widget.set(default)
            else:
                widget = ttk.Entry(ssp_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=2*i+2, padx=2)
            ssp_widgets.append(widget)
            if param == 'id':
                ssp_id_widget = widget
                widget.bind('<KeyRelease>', update_ids)

        # SFH settings
        sfh_frame = ttk.Frame(instance_frame)
        sfh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sfh_frame, text="SFH:").grid(row=0, column=0, sticky=tk.W)
        
        sfh_params = [
            ("id", str(new_id), 5),
            ("itype_sfh", "2", 5, [
                "0: Instantaneous burst",
                "1: Constant",
                "2: Exponentially declining",
                "3: Exponentially increasing",
                "4: Single burst of length tau",
                "5: Delayed",
                "6: Beta",
                "7: Lognormal",
                "8: double power-law",
                "9: Nonparametric"
            ]),
            ("itruncated", "0", 5, ["0: Not truncated", "1: Truncated"]),
            ("itype_ceh", "0", 5, ["0: No CEH", "1: linear mapping model"]),
            ("np_prior_type", "5", 5),
            ("np_interp_method", "0", 5),
            ("np_num_bins", "10", 5),
            ("np_regul", "100", 5)
        ]
        
        sfh_widgets = []
        
        for i, param_info in enumerate(sfh_params):
            param, default, width = param_info[:3]
            ttk.Label(sfh_frame, text=f"{param}:").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(sfh_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
            else:
                widget = ttk.Entry(sfh_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=2*i+2, padx=2)
            sfh_widgets.append(widget)
            if param == 'id':
                sfh_id_widget = widget
                sfh_id_widget.config(state='readonly')
            else:
                widget.config(state='normal')  # Ensure other widgets are editable

        # Add tooltips for SFH parameters
        sfh_tooltips = {
            "id": "Unique ID for the SFH component",
            "itype_sfh": "SFH type (0-9)\n0: Instantaneous burst\n1: Constant\n2: Exponentially declining\n3: Exponentially increasing\n4: Single burst of length tau\n5: Delayed\n6: Beta\n7: Lognormal\n8: double power-law\n9: Nonparametric",
            "itruncated": "Truncation flag (0: Not truncated, 1: Truncated)",
            "itype_ceh": "Chemical evolution history type (0: No CEH, 1: linear mapping model)",
            "np_prior_type": "Prior type for nonparametric SFH (0-7)",
            "np_interp_method": "Interpolation method for nonparametric SFH (0-3)",
            "np_num_bins": "Number of bins for nonparametric SFH",
            "np_regul": "Regularization parameter for nonparametric SFH"
        }
        for i, param_info in enumerate(sfh_params):
            param = param_info[0]
            if param in sfh_tooltips:
                CreateToolTip(sfh_widgets[i], sfh_tooltips[param])

        # Add a trace to the itype_sfh widget to toggle the np_sfh parameters
        def toggle_np_sfh_params(*args):
            itype_sfh = int(sfh_widgets[1].get())
            state = "normal" if itype_sfh == 9 else "disabled"
            for widget in sfh_widgets[4:]:  # NP_SFH parameters start from index 4
                widget.config(state=state)
                widget.config(foreground="black" if state == "normal" else "grey")

        sfh_widgets[1].bind("<<ComboboxSelected>>", toggle_np_sfh_params)

        # Call the function initially to set the correct state
        toggle_np_sfh_params()

        # Create instance-specific BooleanVar variables
        use_dal = tk.BooleanVar(value=False)
        use_dem = tk.BooleanVar(value=False)
        use_kin = tk.BooleanVar(value=False)

        # DAL settings
        dal_frame = ttk.Frame(instance_frame)
        dal_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(dal_frame, text="DAL:", variable=use_dal, 
                        command=lambda: self.toggle_component(dal_params_frame, use_dal.get())).grid(row=0, column=0, sticky='w')

        dal_params_frame = ttk.Frame(dal_frame)
        dal_params_frame.grid(row=0, column=1, sticky='ew')
        dal_params_frame.grid_remove()  # Initially hide the frame

        dal_params = [
            ("id", str(new_id), 5),
            ("con_eml_tot", "2", 5, [
                "0: Continuum",
                "1: Emission lines",
                "2: Total"
            ]),
            ("ilaw", "8", 5, [
                "0: SED model id attenuated and normalized with L_dust",
                "1: Starburst (Calzetti+2000, FAST)",
                "2: Milky Way (Cardelli+1989, FAST)",
                "3: Star-forming (Salim+2018)",
                "4: MW (Allen+76, hyperz)",
                "5: MW (Fitzpatrick+86, hyperz)",
                "6: LMC (Fitzpatrick+86, hyperz)",
                "7: SMC (Fitzpatrick+86, hyperz)",
                "8: SB (Calzetti2000, hyperz)",
                "9: Star-forming (Reddy+2015)"
            ])
        ]
        
        dal_widgets = []
        for i, param_info in enumerate(dal_params):
            param, default, width = param_info[:3]
            ttk.Label(dal_params_frame, text=f"{param}:").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(dal_params_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
            else:
                widget = ttk.Entry(dal_params_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=2*i+2, padx=2)
            dal_widgets.append(widget)
            if param == 'id':
                dal_id_widget = widget
                dal_id_widget.config(state='readonly')

        # Add tooltip for DAL checkbox
        CreateToolTip(dal_frame.winfo_children()[0], "Dust Attenuation Law")

        # DEM settings
        dem_frame = ttk.Frame(instance_frame)
        dem_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Checkbutton(dem_frame, text="DEM:", variable=use_dem, 
                        command=lambda: self.toggle_component(dem_params_frame, use_dem.get())).grid(row=0, column=0, sticky='w')

        dem_params_frame = ttk.Frame(dem_frame)
        dem_params_frame.grid(row=0, column=1, sticky='ew')
        dem_params_frame.grid_remove()  # Initially hide the frame

        dem_params = [
            ("id", str(new_dem_id), 5),  # DEM ID is always one more than the main ID
            ("imodel", "0", 5, ["0: Greybody", "1: Blackbody", "2: FANN", "3: AKNN"]),
            ("iscalable", "-2", 5),
            ("name", "", 15),
        ]

        dem_widgets = []
        for i, param_info in enumerate(dem_params):
            param, default, width = param_info[:3]
            ttk.Label(dem_params_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=2)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(dem_params_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
                if param == "imodel":
                    widget.bind("<<ComboboxSelected>>", lambda event, f=instance_frame: self.update_dem_params(event, f))
            else:
                widget = ttk.Entry(dem_params_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=i*2+1, padx=2)
            dem_widgets.append(widget)
            if param == 'id':
                dem_id_widget = widget
                dem_id_widget.config(state='readonly')

        # Additional parameters for each model type
        self.additional_dem_params = {
            "0": [("ithick", "0", 3), ("w_min", "1", 5), ("w_max", "1000", 5), ("Nw", "200", 5)],  # Greybody
            "1": [("w_min", "1", 5), ("w_max", "1000", 5), ("Nw", "200", 5)],  # Blackbody
            "2": [],  # FANN (no additional parameters)
            "3": [("k", "1", 3), ("f_run", "1", 3), ("eps", "0", 5), ("iRad", "0", 3), 
                  ("iprep", "0", 3), ("Nstep", "1", 3), ("alpha", "0", 5)]  # AKNN
        }

        self.additional_dem_widgets = {}
        for model, params in self.additional_dem_params.items():
            model_widgets = []
            for j, (param, default, width) in enumerate(params):
                label = ttk.Label(dem_params_frame, text=f"{param}:")
                label.grid(row=0, column=len(dem_params)*2 + j*2, sticky=tk.W, padx=2)
                widget = ttk.Entry(dem_params_frame, width=width)
                widget.insert(0, default)
                widget.grid(row=0, column=len(dem_params)*2 + j*2 + 1, padx=2)
                model_widgets.append((label, widget))
            self.additional_dem_widgets[model] = model_widgets

        # Initially hide all additional parameters and their labels
        for widgets in self.additional_dem_widgets.values():
            for label, widget in widgets:
                label.grid_remove()
                widget.grid_remove()

        # Show the initial model's widgets and labels (Greybody by default)
        for label, widget in self.additional_dem_widgets["0"]:
            label.grid()
            widget.grid()

        # Add tooltip for DEM checkbox
        CreateToolTip(dem_frame.winfo_children()[0], "Dust Emission Model")

        # KIN settings
        kin_frame = ttk.Frame(instance_frame)
        kin_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(kin_frame, text="KIN:", variable=use_kin, 
                        command=lambda: self.toggle_component(kin_params_frame, use_kin.get())).grid(row=0, column=0, sticky='w')

        kin_params_frame = ttk.Frame(kin_frame)
        kin_params_frame.grid(row=0, column=1, sticky='ew')
        kin_params_frame.grid_remove()  # Initially hide the frame

        kin_params = [
            ("id", 5),
            ("velscale", 5),
            ("gh_con", 5),
            ("gh_eml", 5)
        ]
        
        kin_widgets = {}
        for i, (param, width) in enumerate(kin_params):
            ttk.Label(kin_params_frame, text=f"{param}:").grid(row=0, column=i*2+1, sticky=tk.W, padx=2)
            widget = ttk.Entry(kin_params_frame, width=width)
            widget.grid(row=0, column=i*2+2, sticky=tk.W, padx=2)
            kin_widgets[param] = widget

        # Set default values and link ID
        kin_widgets['id'].insert(0, ssp_id_widget.get())  # Use the same ID as SSP
        kin_widgets['id'].config(state='readonly')
        kin_widgets['velscale'].insert(0, "10")
        kin_widgets['gh_con'].insert(0, "0")
        kin_widgets['gh_eml'].insert(0, "0")

        # Add tooltips for KIN parameters
        kin_tooltips = {
            "id": "ID of the model (same as SSP, SFH, DAL)",
            "velscale": "Velocity scale (km/s)",
            "gh_con": "Number of Gauss-Hermite terms for continuum",
            "gh_eml": "Number of Gauss-Hermite terms for emission lines"
        }
        for param, tooltip in kin_tooltips.items():
            CreateToolTip(kin_widgets[param], tooltip)

        # Add tooltip for KIN checkbox
        CreateToolTip(kin_frame.winfo_children()[0], "The Stellar and Gas Kinematics")

        # Create the instance dictionary
        new_instance = {
            'frame': instance_frame,
            'ssp': ssp_widgets,
            'sfh': sfh_widgets,
            'dal': dal_widgets,
            'dem': dem_widgets,
            'kin': kin_widgets,
            'ssp_id': ssp_id_widget,
            'sfh_id': sfh_id_widget,
            'dal_id': dal_id_widget,
            'dem_id': dem_id_widget,
            'kin_id': kin_widgets['id'],
            'use_dal': use_dal,
            'use_dem': use_dem,
            'use_kin': use_kin
        }

        # Append the new instance to the list
        self.galaxy_instances.append(new_instance)

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda cf=instance_frame: self.delete_galaxy_instance(cf))
        delete_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # No need to renumber instances, as each instance now has a unique id

    def create_advanced_tab(self):
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Advanced Settings")

        # MultiNest Settings
        multinest_frame, multinest_columns = self.create_advanced_section(advanced_frame, "MultiNest", self.use_multinest, 5)
        multinest_params = [
            ("INS", "Importance Nested Sampling flag (0 or 1)", "1"),
            ("mmodal", "Multimodal flag (0 or 1)", "0"),
            ("ceff", "Constant efficiency mode flag (0 or 1)", "0"),
            ("nlive", "Number of live points", "100"),
            ("efr", "Sampling efficiency", "0.1"),
            ("tol", "Tolerance for termination", "0.5"),
            ("updInt", "Update interval for posterior output", "1000"),
            ("Ztol", "Evidence tolerance", "-1e90"),
            ("seed", "Random seed (0 for system time)", "1"),
            ("fb", "Feedback level (0-3)", "0"),
            ("resume", "Resume from a previous run (0 or 1)", "0"),
            ("outfile", "Write output files (0 or 1)", "0"),
            ("logZero", "Log of Zero (points with loglike < logZero will be ignored)", "-1e90"),
            ("maxiter", "Maximum number of iterations", "100000"),
            ("acpt", "Acceptance rate", "0.01")
        ]
        self.multinest_widgets = self.create_param_widgets(multinest_frame, multinest_params, multinest_columns)

        # NNLM Settings
        nnlm_frame, nnlm_columns = self.create_advanced_section(advanced_frame, "NNLM", self.use_nnlm, 4)
        nnlm_params = [
            ("method", "Method (0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl)", "0"),
            ("Niter1", "Number of iterations for first step", "10000"),
            ("tol1", "Tolerance for first step", "0"),
            ("Niter2", "Number of iterations for second step", "10"),
            ("tol2", "Tolerance for second step", "0.01"),
            ("p1", "Parameter p1 for NNLM algorithm", "0.05"),
            ("p2", "Parameter p2 for NNLM algorithm", "0.95")
        ]
        self.nnlm_widgets = self.create_param_widgets(nnlm_frame, nnlm_params, nnlm_columns)

        # Ndumper Settings
        ndumper_frame, ndumper_columns = self.create_advanced_section(advanced_frame, "Ndumper", self.use_ndumper, 3)
        ndumper_params = [
            ("max_number", "Maximum number of samples to dump", "1"),
            ("iconverged_min", "Minimum convergence flag", "0"),
            ("Xmin_squared_Nd", "Xmin^2/Nd value (-1 for no constraint)", "-1")
        ]
        self.ndumper_widgets = self.create_param_widgets(ndumper_frame, ndumper_params, ndumper_columns)

        # GSL Settings
        gsl_frame, gsl_columns = self.create_advanced_section(advanced_frame, "GSL", self.use_gsl, 3)
        gsl_params = [
            ("integration_epsabs", "Absolute error for GSL integration", "0"),
            ("integration_epsrel", "Relative error for GSL integration", "0.1"),
            ("integration_limit", "Limit for GSL integration", "1000"),
            ("multifit_type", "Multifit type (ols or huber)", "ols"),
            ("multifit_tune", "Tuning parameter for robust fitting", "1.0")
        ]
        self.gsl_widgets = self.create_param_widgets(gsl_frame, gsl_params, gsl_columns)

        # Other Miscellaneous Settings
        misc_frame, misc_columns = self.create_advanced_section(advanced_frame, "Other", self.use_misc, 3)
        misc_params = [
            ("NfilterPoints", "Number of filter points for interpolation", "30"),
            ("Nsample", "Number of samples for catalog creation or SED library building", ""),
            ("niteration", "Number of iterations", "0"),
            ("logZero", "Log of Zero (points with loglike < logZero will be ignored)", "-1e90"),
            ("lw_max", "Max line coverage in km/s for emission line model creation", "10000"),
            ("cl", "Confidence levels for output estimates", "0.68,0.95")
        ]
        self.misc_widgets = self.create_param_widgets(misc_frame, misc_params, misc_columns)

        # Initialize widget states
        for widgets in [self.multinest_widgets, self.nnlm_widgets, self.ndumper_widgets, self.gsl_widgets, self.misc_widgets]:
            self.toggle_widgets(widgets.values(), False)

    def create_advanced_section(self, parent, title, variable, columns):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(frame, variable=variable, 
                        command=lambda: self.toggle_widgets(frame.winfo_children()[1].winfo_children(), variable.get())).pack(side=tk.LEFT, padx=5)
        
        content = ttk.LabelFrame(frame, text=f"{title} Settings")
        content.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        content.columnconfigure(tuple(range(columns * 2)), weight=1)
        
        return content, columns  # Return both the content frame and the number of columns

    def create_param_widgets(self, parent, params, columns):
        widgets = {}
        for i, (param, tooltip, default) in enumerate(params):
            row, col = divmod(i, columns)
            ttk.Label(parent, text=f"{param}:").grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(parent, width=10)
            widget.insert(0, default)
            widget.grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
            widgets[param] = widget
            CreateToolTip(widget, tooltip)
        return widgets

    def create_AGN_tab(self):
        agn_frame = ttk.Frame(self.notebook)
        self.notebook.add(agn_frame, text="AGN")

        # Add button at the top
        ttk.Button(agn_frame, text="Add", command=self.add_AGN_instance).pack(pady=5, anchor=tk.NW)

        # Container for AGN instances
        self.agn_instances_frame = ttk.Frame(agn_frame)
        self.agn_instances_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.agn_instances = []

    def add_AGN_instance(self):
        # Find the maximum ID and igroup among existing galaxy and AGN instances
        max_id = max_igroup = -1
        for instance in self.galaxy_instances:
            max_id = max(max_id, int(instance['ssp'][1].get()))  # id is the second element in ssp list
            max_id = max(max_id, int(instance['dem'][1].get()))  # Consider DEM ID as well
            max_igroup = max(max_igroup, int(instance['ssp'][0].get()))  # igroup is the first element in ssp list
        for instance in self.agn_instances:
            max_id = max(max_id, int(instance['agn_id'].get()))
            max_igroup = max(max_igroup, int(instance['agn_igroup'].get()))
        
        new_id = max_id + 2  # Increment by 2 to ensure uniqueness
        new_igroup = max_igroup + 1  # Increment igroup by 5 for a new AGN instance
        if len(self.agn_instances) > 0:
            new_id = new_id + 4
            new_igroup = new_igroup + 5
        
        instance_frame = ttk.LabelFrame(self.agn_instances_frame, text=f"AGN {len(self.agn_instances)}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create a dictionary to store the BooleanVars for each component
        component_vars = {
            'main_agn': tk.BooleanVar(value=False),
            'bbb': tk.BooleanVar(value=False),
            'blr': tk.BooleanVar(value=False),
            'feii': tk.BooleanVar(value=False),
            'nlr': tk.BooleanVar(value=False),
            'tor': tk.BooleanVar(value=False)
        }

        # Main AGN component
        main_agn_frame = ttk.Frame(instance_frame)
        main_agn_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(main_agn_frame, text="Main", variable=component_vars['main_agn'], 
                        command=lambda: self.toggle_component(agn_params_frame, component_vars['main_agn'].get())).grid(row=0, column=0, sticky='w')

        agn_params_frame = ttk.Frame(main_agn_frame)
        agn_params_frame.grid(row=0, column=1, sticky='ew')

        # Initialize AGN parameters with reduced widths
        agn_igroup = ttk.Entry(agn_params_frame, width=5)
        agn_id = ttk.Entry(agn_params_frame, width=5)
        agn_name = ttk.Entry(agn_params_frame, width=10)
        agn_scalable = ttk.Combobox(agn_params_frame, values=["0", "1"], width=3)
        agn_imodel = ttk.Combobox(agn_params_frame, values=["0", "1", "2", "3", "4", "5"], width=3)
        agn_icloudy = ttk.Combobox(agn_params_frame, values=["0", "1"], width=3)
        agn_suffix = ttk.Entry(agn_params_frame, width=8)
        agn_w_min = ttk.Entry(agn_params_frame, width=6)
        agn_w_max = ttk.Entry(agn_params_frame, width=6)
        agn_nw = ttk.Entry(agn_params_frame, width=5)

        # Layout AGN parameters in a single row
        ttk.Label(agn_params_frame, text="igroup:").grid(row=0, column=0, sticky=tk.W, padx=(0,2))
        agn_igroup.grid(row=0, column=1, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="id:").grid(row=0, column=2, sticky=tk.W, padx=(0,2))
        agn_id.grid(row=0, column=3, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="name:").grid(row=0, column=4, sticky=tk.W, padx=(0,2))
        agn_name.grid(row=0, column=5, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="iscalable:").grid(row=0, column=6, sticky=tk.W, padx=(0,2))
        agn_scalable.grid(row=0, column=7, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="imodel:").grid(row=0, column=8, sticky=tk.W, padx=(0,2))
        agn_imodel.grid(row=0, column=9, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="icloudy:").grid(row=0, column=10, sticky=tk.W, padx=(0,2))
        agn_icloudy.grid(row=0, column=11, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="suffix:").grid(row=0, column=12, sticky=tk.W, padx=(0,2))
        agn_suffix.grid(row=0, column=13, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="w_min:").grid(row=0, column=14, sticky=tk.W, padx=(0,2))
        agn_w_min.grid(row=0, column=15, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="w_max:").grid(row=0, column=16, sticky=tk.W, padx=(0,2))
        agn_w_max.grid(row=0, column=17, sticky=tk.W, padx=(0,5))
        ttk.Label(agn_params_frame, text="Nw:").grid(row=0, column=18, sticky=tk.W, padx=(0,2))
        agn_nw.grid(row=0, column=19, sticky=tk.W, padx=(0,5))

        # Set default values and add tooltips
        agn_igroup.insert(0, str(new_igroup))
        agn_id.insert(0, str(new_id))
        agn_name.insert(0, "AGN")
        agn_scalable.set("1")
        agn_imodel.set("0")
        agn_icloudy.set("0")
        agn_suffix.insert(0, "disk")
        agn_w_min.insert(0, "300.0")
        agn_w_max.insert(0, "1000.0")
        agn_nw.insert(0, "200")

        CreateToolTip(agn_igroup, "Group ID for the AGN component")
        CreateToolTip(agn_id, "Unique ID for the AGN component")
        CreateToolTip(agn_name, "Name of the AGN component")
        CreateToolTip(agn_scalable, "Whether the component is scalable (0: No, 1: Yes)")
        CreateToolTip(agn_imodel, "AGN model type (0: qsosed, 1: agnsed, 2: fagnsed, 3: relagn, 4: relqso, 5: agnslim)")
        CreateToolTip(agn_icloudy, "Whether to use Cloudy model (0: No, 1: Yes)")
        CreateToolTip(agn_suffix, "Suffix for the AGN component name")
        CreateToolTip(agn_w_min, "Minimum wavelength (in microns)")
        CreateToolTip(agn_w_max, "Maximum wavelength (in microns)")
        CreateToolTip(agn_nw, "Number of wavelength points")

        # BBB component
        bbb_frame = ttk.Frame(instance_frame)
        bbb_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(bbb_frame, text="BBB", variable=component_vars['bbb'], 
                        command=lambda: self.toggle_component(bbb_content_frame, component_vars['bbb'].get())).grid(row=0, column=0, sticky='w')

        bbb_content_frame = ttk.Frame(bbb_frame)
        bbb_content_frame.grid(row=0, column=1, sticky='ew')

        ttk.Label(bbb_content_frame, text="igroup:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        bbb_igroup = ttk.Entry(bbb_content_frame, width=5)
        bbb_igroup.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        bbb_igroup.insert(0, str(new_igroup + 1))  # BBB igroup is main AGN igroup + 1
        CreateToolTip(bbb_igroup, "Group ID for the BBB component")

        ttk.Label(bbb_content_frame, text="id:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        bbb_id = ttk.Entry(bbb_content_frame, width=5)
        bbb_id.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        bbb_id.insert(0, str(int(agn_id.get()) + 1))
        CreateToolTip(bbb_id, "Unique ID for the BBB component")

        ttk.Label(bbb_content_frame, text="name:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        bbb_name = ttk.Entry(bbb_content_frame, width=10)
        bbb_name.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        bbb_name.insert(0, "bbb")
        CreateToolTip(bbb_name, "Name of the BBB component")

        ttk.Label(bbb_content_frame, text="w_min:").grid(row=0, column=6, sticky=tk.W, padx=5, pady=2)
        bbb_w_min = ttk.Entry(bbb_content_frame, width=8)
        bbb_w_min.grid(row=0, column=7, sticky=tk.W, padx=5, pady=2)
        bbb_w_min.insert(0, "0.1")
        CreateToolTip(bbb_w_min, "Minimum wavelength for BBB (in microns)")

        ttk.Label(bbb_content_frame, text="w_max:").grid(row=0, column=8, sticky=tk.W, padx=5, pady=2)
        bbb_w_max = ttk.Entry(bbb_content_frame, width=8)
        bbb_w_max.grid(row=0, column=9, sticky=tk.W, padx=5, pady=2)
        bbb_w_max.insert(0, "10")
        CreateToolTip(bbb_w_max, "Maximum wavelength for BBB (in microns)")

        ttk.Label(bbb_content_frame, text="Nw:").grid(row=0, column=10, sticky=tk.W, padx=5, pady=2)
        bbb_nw = ttk.Entry(bbb_content_frame, width=5)
        bbb_nw.grid(row=0, column=11, sticky=tk.W, padx=5, pady=2)
        bbb_nw.insert(0, "1000")
        CreateToolTip(bbb_nw, "Number of wavelength points for BBB")

        # BLR component
        blr_frame = ttk.Frame(instance_frame)
        blr_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(blr_frame, text="BLR", variable=component_vars['blr'], 
                        command=lambda: self.toggle_component(blr_content_frame, component_vars['blr'].get())).grid(row=0, column=0, sticky='w')

        blr_content_frame = ttk.Frame(blr_frame)
        blr_content_frame.grid(row=0, column=1, sticky='ew')

        blr_params = [
            ("igroup", 5), ("id", 5), ("name", 10), ("iscalable", 5),
            ("file", 20), ("R", 5), ("Nsample", 5), ("Nkin", 5)
        ]

        blr_widgets = {}
        for i, (param, width) in enumerate(blr_params):
            ttk.Label(blr_content_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=2)
            widget = ttk.Entry(blr_content_frame, width=width)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=2)
            blr_widgets[param] = widget

        blr_widgets['igroup'].insert(0, str(new_igroup + 2))
        blr_widgets['id'].insert(0, str(int(agn_id.get()) + 2))
        blr_widgets['name'].insert(0, "BLR")
        blr_widgets['iscalable'].insert(0, "1")
        blr_widgets['file'].insert(0, "observation/test/lines_BLR.txt")
        blr_widgets['R'].insert(0, "300")
        blr_widgets['Nsample'].insert(0, "2")
        blr_widgets['Nkin'].insert(0, "3")

        # Add tooltips for BLR parameters
        blr_tooltips = {
            "igroup": "Group ID for the BLR component",
            "id": "Unique ID for the BLR component",
            "name": "Name of the BLR component",
            "iscalable": "Whether the component is scalable (0: No, 1: Yes)",
            "file": "File containing BLR line information",
            "R": "Spectral resolution for BLR",
            "Nsample": "Number of samples for BLR",
            "Nkin": "Number of kinematic components for BLR"
        }
        for param, tooltip in blr_tooltips.items():
            CreateToolTip(blr_widgets[param], tooltip)

        # FeII component
        feii_frame = ttk.Frame(instance_frame)
        feii_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(feii_frame, text="FeII", variable=component_vars['feii'], 
                        command=lambda: self.toggle_component(feii_content_frame, component_vars['feii'].get())).grid(row=0, column=0, sticky='w')

        feii_content_frame = ttk.Frame(feii_frame)
        feii_content_frame.grid(row=0, column=1, sticky='ew')

        # AKNN parameters
        aknn_params = [
            ("igroup", 3), ("id", 3), ("name", 3), ("iscalable", 3),
            ("k", 3), ("f_run", 3), ("eps", 3), ("iRad", 3),
            ("iprep", 3), ("Nstep", 3), ("alpha", 3)
        ]

        feii_widgets = {}
        for i, (param, width) in enumerate(aknn_params):
            ttk.Label(feii_content_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=1)
            widget = ttk.Entry(feii_content_frame, width=width)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=1)
            feii_widgets[param] = widget

        # Set default values
        feii_widgets['igroup'].insert(0, str(new_igroup + 3))
        feii_widgets['id'].insert(0, str(int(agn_id.get()) + 3))
        feii_widgets['name'].insert(0, "FeII")
        feii_widgets['iscalable'].insert(0, "1")
        for param in ['k', 'f_run', 'iprep', 'Nstep', 'alpha']:
            feii_widgets[param].insert(0, "1")
        for param in ['eps', 'iRad']:
            feii_widgets[param].insert(0, "0")

        # Add tooltips for FeII parameters
        feii_tooltips = {
            "igroup": "Group ID for the FeII component",
            "id": "Unique ID for the FeII component",
            "name": "Name of the FeII component",
            "iscalable": "Whether the FeII component is scalable (0: No, 1: Yes)",
            "k": "Number of nearest neighbors for AKNN",
            "f_run": "Fraction of points to use in the running average",
            "eps": "Epsilon parameter for AKNN",
            "iRad": "Radial basis function flag",
            "iprep": "Preprocessing flag",
            "Nstep": "Number of steps for AKNN",
            "alpha": "Alpha parameter for AKNN"
        }
        for param, tooltip in feii_tooltips.items():
            CreateToolTip(feii_widgets[param], tooltip)

        # FeII Kinematic settings
        use_feii_kin = tk.BooleanVar(value=False)
        ttk.Checkbutton(feii_content_frame, text="Kin", variable=use_feii_kin, 
                        command=lambda: self.toggle_widgets(list(kin_widgets.values()), use_feii_kin.get())).grid(row=0, column=len(aknn_params)*2, sticky=tk.W, padx=(5,1))

        kin_params = [("velscale", 3), ("gh_con", 3), ("gh_eml", 3)]
        kin_widgets = {}
        for i, (param, width) in enumerate(kin_params):
            ttk.Label(feii_content_frame, text=f"{param}:").grid(row=0, column=len(aknn_params)*2 + 1 + i*2, sticky=tk.W, padx=1)
            widget = ttk.Entry(feii_content_frame, width=width)
            widget.grid(row=0, column=len(aknn_params)*2 + 2 + i*2, sticky=tk.W, padx=1)
            kin_widgets[param] = widget
            widget.insert(0, "10" if param == "velscale" else "2" if param == "gh_con" else "0")

        # Add tooltips for FeII kinematic parameters
        kin_tooltips = {
            "velscale": "Velocity scale for FeII (km/s)",
            "gh_con": "Number of Gauss-Hermite terms for continuum",
            "gh_eml": "Number of Gauss-Hermite terms for emission"
        }
        for param, tooltip in kin_tooltips.items():
            CreateToolTip(kin_widgets[param], tooltip)

        # Initially disable kinematic widgets
        self.toggle_widgets(list(kin_widgets.values()), False)

        # NLR component
        nlr_frame = ttk.Frame(instance_frame)
        nlr_frame.grid(row=4, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(nlr_frame, text="NLR", variable=component_vars['nlr'], 
                        command=lambda: self.toggle_component(nlr_content_frame, component_vars['nlr'].get())).grid(row=0, column=0, sticky='w')

        nlr_content_frame = ttk.Frame(nlr_frame)
        nlr_content_frame.grid(row=0, column=1, sticky='ew')

        nlr_params = [
            ("igroup", 5), ("id", 5), ("name", 10), ("iscalable", 5),
            ("file", 20), ("R", 5), ("Nsample", 5), ("Nkin", 5)
        ]

        nlr_widgets = {}
        for i, (param, width) in enumerate(nlr_params):
            ttk.Label(nlr_content_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=2)
            widget = ttk.Entry(nlr_content_frame, width=width)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=2)
            nlr_widgets[param] = widget

        nlr_widgets['igroup'].insert(0, str(new_igroup + 4))
        nlr_widgets['id'].insert(0, str(int(agn_id.get()) + 4))
        nlr_widgets['name'].insert(0, "NLR")
        nlr_widgets['iscalable'].insert(0, "1")
        nlr_widgets['file'].insert(0, "observation/test/lines_NLR.txt")
        nlr_widgets['R'].insert(0, "2000")
        nlr_widgets['Nsample'].insert(0, "2")
        nlr_widgets['Nkin'].insert(0, "2")

        # Add tooltips for NLR parameters
        nlr_tooltips = {
            "igroup": "Group ID for the NLR component",
            "id": "Unique ID for the NLR component",
            "name": "Name of the NLR component",
            "iscalable": "Whether the component is scalable (0: No, 1: Yes)",
            "file": "File containing NLR line information",
            "R": "Spectral resolution for NLR",
            "Nsample": "Number of samples for NLR",
            "Nkin": "Number of kinematic components for NLR"
        }
        for param, tooltip in nlr_tooltips.items():
            CreateToolTip(nlr_widgets[param], tooltip)

        # TOR component
        tor_frame = ttk.Frame(instance_frame)
        tor_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(tor_frame, text="TOR", variable=component_vars['tor'], 
                        command=lambda: self.toggle_component(tor_content_frame, component_vars['tor'].get())).grid(row=0, column=0, sticky='w')

        tor_content_frame = ttk.Frame(tor_frame)
        tor_content_frame.grid(row=0, column=1, sticky='ew')

        tor_params = [
            ("igroup", 3), ("id", 3), ("name", 8), ("iscalable", 3),
            ("model_type", 4), ("k", 3), ("f_run", 3), ("eps", 3),
            ("iRad", 3), ("iprep", 3), ("Nstep", 3), ("alpha", 3)
        ]

        tor_widgets = {}
        for i, (param, width) in enumerate(tor_params):
            ttk.Label(tor_content_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=1)
            if param == "model_type":
                widget = ttk.Combobox(tor_content_frame, values=["FANN", "AKNN"], width=width)
                widget.set("FANN")
                widget.bind("<<ComboboxSelected>>", lambda e, widgets=tor_widgets: self.toggle_tor_params(widgets))
            else:
                widget = ttk.Entry(tor_content_frame, width=width)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=1)
            tor_widgets[param] = widget

        # Set default values for TOR widgets
        tor_widgets['igroup'].insert(0, str(new_igroup + 5))
        tor_widgets['id'].insert(0, str(int(agn_id.get()) + 5))
        tor_widgets['name'].insert(0, "clumpy201410tor")
        tor_widgets['iscalable'].insert(0, "1")
        tor_widgets['k'].insert(0, "1")
        tor_widgets['f_run'].insert(0, "1")
        tor_widgets['eps'].insert(0, "0")
        tor_widgets['iRad'].insert(0, "0")
        tor_widgets['iprep'].insert(0, "0")
        tor_widgets['Nstep'].insert(0, "1")
        tor_widgets['alpha'].insert(0, "0")

        # Add tooltips for TOR parameters
        tor_tooltips = {
            "igroup": "Group ID for the TOR component",
            "id": "Unique ID for the TOR component",
            "name": "Name of the TOR component",
            "iscalable": "Whether the component is scalable (0: No, 1: Yes)",
            "model_type": "Type of TOR model (FANN or AKNN)",
            "k": "Number of nearest neighbors (for AKNN)",
            "f_run": "Fraction of points to use in the running average",
            "eps": "Epsilon parameter (for AKNN)",
            "iRad": "Radial basis function flag (for AKNN)",
            "iprep": "Preprocessing flag (for AKNN)",
            "Nstep": "Number of steps (for AKNN)",
            "alpha": "Alpha parameter (for AKNN)"
        }
        for param, tooltip in tor_tooltips.items():
            CreateToolTip(tor_widgets[param], tooltip)

        # Initialize the TOR parameters visibility
        self.toggle_tor_params(tor_widgets)

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda: self.delete_AGN_instance(instance_frame))
        delete_button.grid(row=6, column=0, sticky='e', padx=5, pady=5)

        # Update the instance dictionary
        new_instance = {
            'frame': instance_frame,
            'component_vars': component_vars,
            'agn_igroup': agn_igroup,
            'agn_id': agn_id,
            'name': agn_name,
            'iscalable': agn_scalable,
            'imodel': agn_imodel,
            'icloudy': agn_icloudy,
            'suffix': agn_suffix,
            'w_min': agn_w_min,
            'w_max': agn_w_max,
            'nw': agn_nw,
            'bbb_frame': bbb_frame,
            'bbb_igroup': bbb_igroup,
            'bbb_id': bbb_id,
            'bbb_name': bbb_name,
            'bbb_w_min': bbb_w_min,
            'bbb_w_max': bbb_w_max,
            'bbb_nw': bbb_nw,
            'blr_frame': blr_frame,
            'blr_widgets': blr_widgets,
            'feii_frame': feii_frame,
            'feii_widgets': feii_widgets,
            'kin_widgets': kin_widgets,
            'nlr_frame': nlr_frame,
            'nlr_widgets': nlr_widgets,
            'tor_frame': tor_frame,
            'tor_widgets': tor_widgets,
            'use_feii_kin': use_feii_kin
        }

        self.agn_instances.append(new_instance)

        # Initialize the component visibilities
        for component, var in component_vars.items():
            if component == 'main_agn':
                self.toggle_component(agn_params_frame, var.get())
            elif component == 'tor':
                self.toggle_component(tor_content_frame, var.get())
            else:
                self.toggle_component(locals()[f"{component}_content_frame"], var.get())

        # Configure column weights to allow expansion
        instance_frame.grid_columnconfigure(0, weight=1)
        main_agn_frame.grid_columnconfigure(1, weight=1)
        bbb_frame.grid_columnconfigure(1, weight=1)
        blr_frame.grid_columnconfigure(1, weight=1)
        feii_frame.grid_columnconfigure(1, weight=1)
        nlr_frame.grid_columnconfigure(1, weight=1)
        tor_frame.grid_columnconfigure(1, weight=1)

    # Add this new method to toggle component visibility
    def toggle_component(self, frame, state):
        if frame is None:
            print(f"Warning: Frame is None. Skipping.")
            return
        
        if state:
            frame.grid()
        else:
            frame.grid_remove()

        for child in frame.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Combobox)):
                child.config(state="normal" if state else "disabled")

    def get_agn_settings(self):
        return [
            {key: (widget.get() if isinstance(widget, (ttk.Entry, ttk.Combobox)) else 
                   widget.get() if isinstance(widget, tk.BooleanVar) else 
                   {k: v.get() if hasattr(v, 'get') else v for k, v in widget.items()} if isinstance(widget, dict) else 
                   widget)
             for key, widget in instance.items() if key not in ['frame', 'bbb_frame', 'blr_frame', 'nlr_frame', 'feii_frame', 'tor_frame']}
            for instance in self.agn_instances
        ]

    def toggle_tor_params(self, tor_widgets):
        model_type = tor_widgets['model_type'].get()
        common_params = ['igroup', 'id', 'name', 'iscalable', 'model_type']
        aknn_params = ['k', 'f_run', 'eps', 'iRad', 'iprep', 'Nstep', 'alpha']
        
        for i, (param, widget) in enumerate(tor_widgets.items()):
            try:
                label_widget = widget.master.grid_slaves(row=0, column=i*2)[0]
            except (KeyError, IndexError):
                # If we can't find the label widget, we'll create a new one
                label_widget = ttk.Label(widget.master, text=f"{param}:")
                label_widget.grid(row=0, column=i*2, sticky=tk.W, padx=1)
                widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=1)

            if param in common_params:
                widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=1)
                widget.config(state="normal")
                label_widget.grid(row=0, column=i*2, sticky=tk.W, padx=1)
            elif param in aknn_params:
                if model_type == "AKNN":
                    widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=1)
                    widget.config(state="normal")
                    label_widget.grid(row=0, column=i*2, sticky=tk.W, padx=1)
                else:
                    widget.grid_remove()
                    label_widget.grid_remove()

        # Adjust the layout after toggling
        tor_widgets['model_type'].master.update_idletasks()

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)

    def run_bayesed(self):
        if self.run_button['text'] == "Run":
            params = self.create_bayesed_params()
            
            np = self.mpi_processes.get().strip()
            ntest = self.ntest.get().strip()
            
            self.output_queue = queue.Queue()
            self.stop_output_thread = threading.Event()
            threading.Thread(target=self.execute_bayesed, args=(params, np, ntest), daemon=True).start()
            self.master.after(100, self.check_output_queue)

            # Change button text to "Stop"
            self.run_button.config(text="Stop", command=self.stop_bayesed)
        else:
            self.stop_bayesed()

    def execute_bayesed(self, params, np, ntest):
        try:
            bayesed = BayeSEDInterface(mpi_mode='1', np=int(np) if np else None, Ntest=int(ntest) if ntest else None)
            
            # Show the full command in the output box
            self.output_queue.put("Executing BayeSED...\n")
            
            # Run BayeSED
            bayesed.run(params)
            
            self.output_queue.put("BayeSED execution completed\n")
            
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}\n")
        
        finally:
            self.output_queue.put(None)  # Signal that the process has finished
            # Change button text back to "Run"
            self.master.after(0, lambda: self.run_button.config(text="Run", command=self.run_bayesed))

    def create_bayesed_params(self):
        params = BayeSEDParams(
            input_type=int(self.input_type.get().split()[0]),
            input_file=self.input_file.get(),
            outdir=self.outdir.get(),
            verbose=int(self.verbose.get()),
            save_bestfit=int(self.save_bestfit_type.get().split()[0]) if self.save_bestfit.get() else 0,
            save_sample_par=self.save_sample_par.get(),
            save_sample_obs=self.save_sample_obs.get(),
            filters=self.filters.get() or None,
            filters_selected=self.filters_selected.get() or None,
            no_photometry_fit=self.no_photometry_fit.get(),
            no_spectra_fit=self.no_spectra_fit.get(),
            unweighted_samples=self.unweighted_samples.get(),
            priors_only=self.priors_only.get(),
        )

        # Initialize all list parameters as empty lists
        params.ssp = []
        params.sfh = []
        params.dal = []
        params.greybody = []
        params.blackbody = []
        params.fann = []
        params.aknn = []
        params.kin = []
        params.AGN = []
        params.big_blue_bump = []
        params.lines1 = []  # This includes both BLR and NLR

        # Galaxy instances
        for instance in self.galaxy_instances:
            ssp_values = [widget.get() for widget in instance['ssp']]
            params.ssp.append(SSPParams(
                igroup=int(ssp_values[0]),
                id=int(ssp_values[1]),
                name=ssp_values[2],
                iscalable=int(ssp_values[3]),
                k=int(ssp_values[4]),
                f_run=int(ssp_values[5]),
                Nstep=int(ssp_values[6]),
                i0=int(ssp_values[7]),
                i1=int(ssp_values[8]),
                i2=int(ssp_values[9]),
                i3=int(ssp_values[10])
            ))

            sfh_values = [widget.get() for widget in instance['sfh']]
            params.sfh.append(SFHParams(
                id=int(sfh_values[0]),
                itype_sfh=int(sfh_values[1]),
                itruncated=int(sfh_values[2]),
                itype_ceh=int(sfh_values[3])
            ))

            if instance['use_dal'].get():
                dal_values = [widget.get() for widget in instance['dal']]
                params.dal.append(DALParams(
                    id=int(dal_values[0]),
                    con_eml_tot=int(dal_values[1]),
                    ilaw=int(dal_values[2])
                ))

            if instance['use_dem'].get():
                dem_values = [widget.get() for widget in instance['dem']]
                imodel = dem_values[1]
                
                # Common parameters for all DEM models
                dem_params = {
                    'igroup': int(dem_values[0]),
                    'id': int(dem_values[0]),  # Using the same value for id and igroup
                    'name': dem_values[3],
                    'iscalable': int(dem_values[2]) if dem_values[2].strip() else -2  # Default to -2 if empty
                }

                if imodel == "0":  # Greybody
                    additional_params = [widget for _, widget in self.additional_dem_widgets[imodel]]
                    params.greybody.append(GreybodyParams(
                        **dem_params,
                        ithick=int(additional_params[0].get() or 0),
                        w_min=float(additional_params[1].get() or 1),
                        w_max=float(additional_params[2].get() or 1000),
                        Nw=int(additional_params[3].get() or 200)
                    ))
                elif imodel == "1":  # Blackbody
                    additional_params = [widget for _, widget in self.additional_dem_widgets[imodel]]
                    params.blackbody.append(BlackbodyParams(
                        **dem_params,
                        w_min=float(additional_params[0].get() or 1),
                        w_max=float(additional_params[1].get() or 1000),
                        Nw=int(additional_params[2].get() or 200)
                    ))
                elif imodel == "2":  # FANN
                    params.fann.append(FANNParams(**dem_params))
                elif imodel == "3":  # AKNN
                    additional_params = [widget for _, widget in self.additional_dem_widgets[imodel]]
                    params.aknn.append(AKNNParams(
                        **dem_params,
                        k=int(additional_params[0].get() or 1),
                        f_run=int(additional_params[1].get() or 1),
                        eps=float(additional_params[2].get() or 0),
                        iRad=int(additional_params[3].get() or 0),
                        iprep=int(additional_params[4].get() or 0),
                        Nstep=int(additional_params[5].get() or 1),
                        alpha=float(additional_params[6].get() or 0)
                    ))

            if instance['use_kin'].get():
                kin_values = [widget.get() for widget in instance['kin'].values()]
                params.kin.append(KinParams(
                    id=int(kin_values[0]),
                    velscale=int(kin_values[1]),
                    num_gauss_hermites_con=int(kin_values[2]),
                    num_gauss_hermites_eml=int(kin_values[3])
                ))

        # AGN instances
        for agn in self.agn_instances:
            if agn['component_vars']['main_agn'].get():
                params.AGN.append(AGNParams(
                    igroup=int(agn['agn_igroup'].get()),
                    id=int(agn['agn_id'].get()),
                    name=agn['name'].get(),
                    iscalable=int(agn['iscalable'].get()),
                    imodel=int(agn['imodel'].get().split()[0]),
                    icloudy=int(agn['icloudy'].get()),
                    suffix=agn['suffix'].get(),
                    w_min=float(agn['w_min'].get()),
                    w_max=float(agn['w_max'].get()),
                    Nw=int(agn['nw'].get())
                ))

            if agn['component_vars']['bbb'].get():
                params.big_blue_bump.append(BigBlueBumpParams(
                    igroup=int(agn['bbb_igroup'].get()),
                    id=int(agn['bbb_id'].get()),
                    name=agn['bbb_name'].get(),
                    iscalable=1,
                    w_min=float(agn['bbb_w_min'].get()),
                    w_max=float(agn['bbb_w_max'].get()),
                    Nw=int(agn['bbb_nw'].get())
                ))

            if agn['component_vars']['blr'].get():
                blr = agn['blr_widgets']
                params.lines1.append(LineParams(
                    igroup=int(blr['igroup'].get()),
                    id=int(blr['id'].get()),
                    name=blr['name'].get(),
                    iscalable=int(blr['iscalable'].get()),
                    file=blr['file'].get(),
                    R=float(blr['R'].get()),
                    Nsample=int(blr['Nsample'].get()),
                    Nkin=int(blr['Nkin'].get())
                ))

            if agn['component_vars']['feii'].get():
                feii = agn['feii_widgets']
                feii_params = AKNNParams(
                    igroup=int(feii['igroup'].get()),
                    id=int(feii['id'].get()),
                    name=feii['name'].get(),
                    iscalable=int(feii['iscalable'].get()),
                    k=int(feii['k'].get()),
                    f_run=int(feii['f_run'].get()),
                    eps=float(feii['eps'].get()),
                    iRad=int(feii['iRad'].get()),
                    iprep=int(feii['iprep'].get()),
                    Nstep=int(feii['Nstep'].get()),
                    alpha=float(feii['alpha'].get())
                )
                params.aknn.append(feii_params)

                # Add FeII kinematic parameters if they are used
                if agn['use_feii_kin'].get():
                    kin_widgets = agn['kin_widgets']
                    params.kin.append(KinParams(
                        id=int(feii['id'].get()),  # Use the same ID as the FeII component
                        velscale=int(kin_widgets['velscale'].get()),
                        num_gauss_hermites_con=int(kin_widgets['gh_con'].get() or 0),
                        num_gauss_hermites_eml=int(kin_widgets['gh_eml'].get() or 0)
                    ))

            if agn['component_vars']['nlr'].get():
                nlr = agn['nlr_widgets']
                params.lines1.append(LineParams(
                    igroup=int(nlr['igroup'].get()),
                    id=int(nlr['id'].get()),
                    name=nlr['name'].get(),
                    iscalable=int(nlr['iscalable'].get()),
                    file=nlr['file'].get(),
                    R=float(nlr['R'].get()),
                    Nsample=int(nlr['Nsample'].get()),
                    Nkin=int(nlr['Nkin'].get())
                ))

            if agn['component_vars']['tor'].get():
                tor = agn['tor_widgets']
                if tor['model_type'].get() == "FANN":
                    params.fann.append(FANNParams(
                        igroup=int(tor['igroup'].get()),
                        id=int(tor['id'].get()),
                        name=tor['name'].get(),
                        iscalable=int(tor['iscalable'].get())
                    ))
                else:  # AKNN
                    params.aknn.append(AKNNParams(
                        igroup=int(tor['igroup'].get()),
                        id=int(tor['id'].get()),
                        name=tor['name'].get(),
                        iscalable=int(tor['iscalable'].get()),
                        k=int(tor['k'].get()),
                        f_run=int(tor['f_run'].get()),
                        eps=float(tor['eps'].get()),
                        iRad=int(tor['iRad'].get()),
                        iprep=int(tor['iprep'].get()),
                        Nstep=int(tor['Nstep'].get()),
                        alpha=float(tor['alpha'].get())
                    ))

        # Advanced settings
        if self.use_multinest.get():
            params.multinest = MultiNestParams(**{param: float(widget.get()) for param, widget in self.multinest_widgets.items()})

        if self.use_nnlm.get():
            params.NNLM = NNLMParams(**{param: float(widget.get()) for param, widget in self.nnlm_widgets.items()})

        if self.use_ndumper.get():
            params.Ndumper = NdumperParams(**{param: float(widget.get()) for param, widget in self.ndumper_widgets.items()})

        if self.use_gsl.get():
            params.gsl_integration_qag = GSLIntegrationQAGParams(
                epsabs=float(self.gsl_widgets['integration_epsabs'].get()),
                epsrel=float(self.gsl_widgets['integration_epsrel'].get()),
                limit=int(self.gsl_widgets['integration_limit'].get())
            )
            params.gsl_multifit_robust = GSLMultifitRobustParams(
                type=self.gsl_widgets['multifit_type'].get(),
                tune=float(self.gsl_widgets['multifit_tune'].get())
            )

        if self.use_misc.get():
            for param, widget in self.misc_widgets.items():
                value = widget.get()
                if value:
                    setattr(params, param, int(value) if param != 'cl' else value)

        # Cosmology settings
        if self.use_cosmology.get():
            params.cosmology = CosmologyParams(
                H0=float(self.cosmology_params['H0'].get()),
                omigaA=float(self.cosmology_params['omigaA'].get()),
                omigam=float(self.cosmology_params['omigam'].get())
            )

        # IGM model
        if self.use_igm.get():
            params.IGM = int(self.igm_model.get())

        # Redshift parameters
        if self.use_redshift.get():
            params.z = ZParams(**{param: float(widget.get()) for param, widget in self.redshift_params.items()})

        # Other settings
        if self.use_sfr.get():
            params.SFR_over = SFROverParams(
                past_Myr1=float(self.sfr_myr_entry.get().split(',')[0]),
                past_Myr2=float(self.sfr_myr_entry.get().split(',')[1])
            )

        if self.use_snr.get():
            params.SNRmin1 = SNRmin1Params(
                phot=float(self.snrmin1.get().split(',')[0]),
                spec=float(self.snrmin1.get().split(',')[1])
            )
            params.SNRmin2 = SNRmin2Params(
                phot=float(self.snrmin2.get().split(',')[0]),
                spec=float(self.snrmin2.get().split(',')[1])
            )

        if self.use_build_sedlib.get():
            params.build_sedlib = int(self.build_sedlib.get().split()[0])

        if self.use_output_sfh.get():
            params.output_SFH = OutputSFHParams(
                ntimes=int(self.output_sfh_ntimes.get()),
                ilog=int(self.output_sfh_ilog.get())
            )

        if self.use_sys_err.get():
            params.sys_err_mod = SysErrParams(**{param: float(widget.get()) for param, widget in zip(['iprior_type', 'is_age', 'min', 'max', 'nbin'], self.sys_err_widgets[:5])})
            params.sys_err_obs = SysErrParams(**{param: float(widget.get()) for param, widget in zip(['iprior_type', 'is_age', 'min', 'max', 'nbin'], self.sys_err_widgets[5:])})

        return params

    def stop_bayesed(self):
        if hasattr(self, 'process') and self.process:
            # Terminate the main Python process
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            
            # If using MPI, terminate all MPI processes
            np = self.mpi_processes.get().strip()
            if np:
                try:
                    # This will terminate all processes with 'mpirun' or 'bayesed' in their command line
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        if 'mpirun' in proc.info['name'] or 'bayesed' in proc.info['name']:
                            proc.terminate()
                except Exception as e:
                    print(f"Error terminating MPI processes: {e}")
            
            # Wait for all processes to actually terminate
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            for p in alive:
                p.kill()  # Force kill if still alive
        
        self.stop_output_thread.set()
        
        # Change button text back to "Run"
        self.run_button.config(text="Run", command=self.run_bayesed)
        
        self.update_output("BayeSED execution stopped by user.\n")

    def execute_command(self, command, np, ntest):
        import sys
        import threading
        
        try:
            # Use subprocess.STARTUPINFO to hide console window on Windows
            startupinfo = None
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            bayesed = BayeSEDInterface(mpi_mode='1', np=int(np) if np else None, Ntest=int(ntest) if ntest else None)
            
            # Prepare the command
            mpi_command = [bayesed.mpi_cmd, '--use-hwthread-cpus']
            if bayesed.np is not None:
                mpi_command.extend(['-np', str(bayesed.np)])
            mpi_command.append(bayesed.executable_path)
            full_command = mpi_command + command
            
            # Show the full command in the output box
            self.output_queue.put("Executing command: " + " ".join(full_command) + "\n")
            
            # Create a thread to read the output in real-time
            def output_reader(stream, queue):
                for line in iter(stream.readline, b''):
                    queue.put(line.decode('utf-8', errors='replace'))  # Preserve all characters, including tabs
                stream.close()
            
            # Start BayeSED process
            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo
            )
            
            # Start output reader thread
            output_thread = threading.Thread(target=output_reader, args=(process.stdout, self.output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Wait for the process to complete
            process.wait()
            
            # Wait for the output thread to finish
            output_thread.join()
            
            if process.returncode == 0:
                self.output_queue.put("BayeSED execution completed\n")
            else:
                self.output_queue.put(f"BayeSED execution failed, return code: {process.returncode}\n")
            
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}\n")
        
        finally:
            self.output_queue.put(None)  # Signal that the process has finished
            # Change button text back to "Run"
            self.master.after(0, lambda: self.run_button.config(text="Run", command=self.run_bayesed))

    def check_output_queue(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line is None:  # Process has finished
                    self.run_button.config(text="Run", command=self.run_bayesed)
                    return
                self.update_output(line)
        except queue.Empty:
            self.master.after(100, self.check_output_queue)

    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        
        # Replace tabs with a fixed number of spaces
        tab_size = 4  # You can adjust this value
        text = text.replace('\t', ' ' * tab_size)
        
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.NORMAL, font=self.standard_font)  # Use the standard font
        self.output_text.update_idletasks()  # Force update of the widget

    def browse_input_file(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename:
            self.input_file.delete(0, tk.END)
            self.input_file.insert(0, os.path.relpath(filename))

    def browse_outdir(self):
        dirname = filedialog.askdirectory(initialdir=os.getcwd())
        if dirname:
            self.outdir.delete(0, tk.END)
            self.outdir.insert(0, os.path.relpath(dirname))

    def create_cosmology_tab(self):
        cosmology_frame = ttk.Frame(self.notebook)
        self.notebook.add(cosmology_frame, text="Cosmology")

        # Cosmology parameters
        cosmo_frame = ttk.Frame(cosmology_frame)
        cosmo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(cosmo_frame, variable=self.use_cosmology, 
                        command=lambda: self.toggle_widgets(list(self.cosmology_params.values()), self.use_cosmology.get())).pack(side=tk.LEFT, padx=5)

        cosmo_content = ttk.LabelFrame(cosmo_frame, text="Cosmology Parameters")
        cosmo_content.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        cosmo_params = [
            ("H0", "Hubble constant (km/s/Mpc)", "70"),
            ("omigaA", "Omega Lambda", "0.7"),
            ("omigam", "Omega Matter", "0.3")
        ]

        for param, tooltip, default in cosmo_params:
            param_frame = ttk.Frame(cosmo_content)
            param_frame.pack(fill=tk.X, pady=2)
            ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT, padx=5)
            widget = ttk.Entry(param_frame, width=10)
            widget.insert(0, default)
            widget.pack(side=tk.LEFT, padx=5)
            self.cosmology_params[param] = widget
            CreateToolTip(widget, tooltip)

        # IGM model
        igm_frame = ttk.Frame(cosmology_frame)
        igm_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(igm_frame, variable=self.use_igm, 
                        command=lambda: self.toggle_widgets(self.igm_radiobuttons, self.use_igm.get())).pack(side=tk.LEFT, padx=5)

        igm_content = ttk.LabelFrame(igm_frame, text="IGM Model")
        igm_content.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.igm_model = tk.StringVar(value="1")
        self.igm_radiobuttons = []
        igm_options = [
            ("0", "None"), ("1", "Madau (1995)"), ("2", "Meiksin (2006)"),
            ("3", "hyperz"), ("4", "FSPS"), ("5", "Inoue+2014")
        ]

        for value, text in igm_options:
            radiobutton = ttk.Radiobutton(igm_content, text=text, variable=self.igm_model, value=value)
            radiobutton.pack(anchor="w", padx=5, pady=2)
            self.igm_radiobuttons.append(radiobutton)

        # Redshift parameters
        redshift_frame = ttk.Frame(cosmology_frame)
        redshift_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(redshift_frame, variable=self.use_redshift, 
                        command=self.toggle_redshift_widgets).pack(side=tk.LEFT, padx=5)

        redshift_content = ttk.LabelFrame(redshift_frame, text="Redshift Parameters")
        redshift_content.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        redshift_params = [
            ("iprior_type", "Prior type (0-7)", "1"),
            ("is_age", "Age-dependent flag (0 or 1)", "0"),
            ("min", "Minimum redshift", "0"),
            ("max", "Maximum redshift", "10"),
            ("nbin", "Number of bins", "100")
        ]

        for param, tooltip, default in redshift_params:
            param_frame = ttk.Frame(redshift_content)
            param_frame.pack(fill=tk.X, pady=2)
            ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT, padx=5)
            widget = ttk.Entry(param_frame, width=10)
            widget.insert(0, default)
            widget.pack(side=tk.LEFT, padx=5)
            self.redshift_params[param] = widget
            CreateToolTip(widget, tooltip)
            self.redshift_widgets.append(widget)

        # Initialize widget states
        self.toggle_widgets(list(self.cosmology_params.values()), False)
        self.toggle_widgets(self.igm_radiobuttons, False)
        self.toggle_redshift_widgets()

    def toggle_widgets(self, widgets, state):
        for widget in widgets:
            if isinstance(widget, ttk.Radiobutton):
                widget.config(state="normal" if state else "disabled")
            elif hasattr(widget, 'config'):
                widget.config(state="normal" if state else "disabled")
                if state:
                    widget.config(foreground="black")
                else:
                    widget.config(foreground="grey")

    def toggle_redshift_widgets(self):
        state = self.use_redshift.get()
        for widget in self.redshift_widgets:
            self.toggle_widgets([widget], state)

    def browse_filters(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename:
            self.filters.delete(0, tk.END)
            self.filters.insert(0, os.path.relpath(filename))

    def browse_filters_selected(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename:
            self.filters_selected.delete(0, tk.END)
            self.filters_selected.insert(0, os.path.relpath(filename))

    def update_dem_params(self, event, frame):
        # Find the correct instance
        instance = next((inst for inst in self.galaxy_instances if inst['frame'] == frame), None)
        if not instance:
            return  # If we can't find the instance, just return

        imodel = event.widget.get() if event else instance['dem'][1].get()

        # Hide all additional parameter widgets and their labels
        for widgets in self.additional_dem_widgets.values():
            for label, widget in widgets:
                label.grid_remove()
                widget.grid_remove()

        # Show the widgets and labels for the selected model
        if imodel in self.additional_dem_widgets:
            for label, widget in self.additional_dem_widgets[imodel]:
                label.grid()
                widget.grid()

        # Force the frame to update its layout
        frame.update_idletasks()

    def delete_galaxy_instance(self, frame):
        for instance in self.galaxy_instances:
            if instance['frame'] == frame:
                # Remove the extra frame if it exists
                if 'dem_extra_frame' in instance:
                    instance['dem_extra_frame'].destroy()
                self.galaxy_instances.remove(instance)
                break
        frame.destroy()

    def delete_AGN_instance(self, frame):
        for instance in self.agn_instances:
            if instance['frame'] == frame:
                self.agn_instances.remove(instance)
                break
        frame.destroy()

    def stop_execution(self):
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
        self.stop_output_thread.set()

    def copy_selection(self, event):
        try:
            selected_text = self.output_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            pyperclip.copy(selected_text)
        except tk.TclError:
            pass  # No selection
        return "break"

    def plot_bestfit(self):
        if not hasattr(self, 'full_fits_path') or not os.path.exists(self.full_fits_path):
            output_dir = self.outdir.get()
            fits_file = self.fits_file.get()
            full_path = os.path.join(output_dir, fits_file)
            
            if not os.path.exists(full_path):
                messagebox.showerror("Error", f"FITS file not found: {full_path}")
                return
        else:
            full_path = self.full_fits_path

        plot_script = "plot/plot_bestfit.py"
        
        # Check if filter files exist
        filter_file = self.filters.get()
        filter_names_file = self.filters_selected.get()

        if filter_file and filter_names_file and os.path.exists(filter_file) and os.path.exists(filter_names_file):
            command = [sys.executable, plot_script, full_path, filter_file, filter_names_file]
        else:
            command = [sys.executable, plot_script, full_path]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to run plot script: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def browse_fits_file(self):
        output_dir = self.outdir.get()
        if not output_dir:
            messagebox.showerror("Error", "Please set an output directory in the input settings first.")
            return
        
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", f"The specified output directory does not exist: {output_dir}")
            return

        filename = filedialog.askopenfilename(
            initialdir=output_dir,
            title="Select FITS file",
            filetypes=(("FITS files", "*.fits"), ("All files", "*.*"))
        )
        if filename:
            # Store the full path, but display only the relative path from the output directory
            self.full_fits_path = filename
            relative_path = os.path.relpath(filename, output_dir)
            self.fits_file.delete(0, tk.END)
            self.fits_file.insert(0, relative_path)

    def save_script(self):
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a default filename with timestamp
        default_filename = f"bayesed_script_{timestamp}.py"
        
        # Open a file dialog to choose where to save the script
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write("from bayesed import BayeSEDInterface, BayeSEDParams\n")
                f.write("from bayesed import FANNParams, AGNParams, BlackbodyParams, BigBlueBumpParams, GreybodyParams\n")
                f.write("from bayesed import AKNNParams, LineParams, LuminosityParams, NPSFHParams, PolynomialParams\n")
                f.write("from bayesed import PowerlawParams, RBFParams, SFHParams, SSPParams, SEDLibParams, SysErrParams\n")
                f.write("from bayesed import ZParams, NNLMParams, NdumperParams, OutputSFHParams, MultiNestParams\n")
                f.write("from bayesed import SFROverParams, SNRmin1Params, SNRmin2Params, GSLIntegrationQAGParams\n")
                f.write("from bayesed import GSLMultifitRobustParams, KinParams, LineListParams, MakeCatalogParams\n")
                f.write("from bayesed import CloudyParams, CosmologyParams, DALParams, RDFParams, TemplateParams\n\n")

                f.write("def run_bayesed():\n")
                
                # Get the number of MPI processes and Ntest
                np = self.mpi_processes.get().strip()
                ntest = self.ntest.get().strip()
                
                # Initialize BayeSEDInterface with np and Ntest
                f.write(f"    bayesed = BayeSEDInterface(mpi_mode='1'")
                if np:
                    f.write(f", np={np}")
                if ntest:
                    f.write(f", Ntest={ntest}")
                f.write(")\n\n")
                
                # Create BayeSEDParams
                params = self.create_bayesed_params()
                
                # Write BayeSEDParams to file
                f.write("    params = BayeSEDParams(\n")
                for field in params.__dataclass_fields__:
                    value = getattr(params, field)
                    if value is not None and value != []:
                        f.write(f"        {field}={repr(value)},\n")
                f.write("    )\n\n")
                
                f.write("    bayesed.run(params)\n")
                f.write("\nif __name__ == '__main__':\n")
                f.write("    run_bayesed()\n")
            
            messagebox.showinfo("Save Successful", f"Script saved to {filename}")

    def create_about_button(self):
        about_button = ttk.Button(self.master, text="About", command=self.show_about_window)
        about_button.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=10)

    def show_about_window(self):
        about_window = tk.Toplevel(self.master)
        about_window.title("About BayeSED3")
        
        window_width, window_height = 1200, 800  # Increased width to 1200 and height to 800
        screen_width, screen_height = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        x, y = (screen_width - window_width) // 2, (screen_height - window_height) // 2
        about_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        main_frame = ttk.Frame(about_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content_frame = ttk.Frame(scrollable_frame)
        content_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Header section
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill="x", pady=10)

        try:
            logo = Image.open("BayeSED3.jpg")
            logo = logo.resize((100, 100), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo)
            logo_label = ttk.Label(header_frame, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack(side="left", padx=(0, 20))
        except Exception as e:
            print(f"Error loading logo: {e}")

        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side="left")
        ttk.Label(title_frame, text="BayeSED3: A code for Bayesian SED synthesis and analysis of galaxies and AGNs", font=("Helvetica", 24, "bold")).pack(anchor="w")
        # ttk.Label(title_frame, text="Version 3.0", font=("Helvetica", 16)).pack(anchor="w")

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Description section
        description = ("BayeSED3 is a general and sophisticated tool for the full Bayesian interpretation (parameter estimation and model comparison) of spectral energy distributions (SEDs).")
        ttk.Label(content_frame, text="Description", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10, 5))
        ttk.Label(content_frame, text=description, wraplength=1160, justify="left").pack(pady=(0, 10))

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Features section
        features = (
            "• Multi-component SED synthesis and analysis for galaxies and AGNs\n"
            "• Flexible stellar population synthesis modeling\n"
            "• Flexible dust attenuation and emission modeling\n"
            "• Flexible stellar and gas kinematics modeling\n"
            "• Non-parametric and parametric star formation history options\n"
            "• Comprehensive AGN component modeling (Accretion disk, BLR, NLR, Torus)\n"
            "• Intergalactic medium (IGM) absorption modeling\n"
            "• Handling of both photometric and spectroscopic data\n"
            "• Bayesian parameter estimation and model comparison\n"
            "• Machine learning techniques for SED model emulation\n"
            "• Parallel processing support for improved performance\n"
            "• User-friendly CLI, python script and GUI interfaces"
        )
        ttk.Label(content_frame, text="Key Features", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10, 5))
        ttk.Label(content_frame, text=features,font=("Helvetica", 16), wraplength=1160, justify="left").pack(pady=(0, 10))

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Technical Details section
        tech_details = (
            "BayeSED3 is built with a hybrid architecture, combining Python's flexibility with C++'s performance. "
            "It leverages OpenMPI for distributed computing, supporting both single-object and multi-object analysis modes. "
            "The software employs the MultiNest algorithm for efficient exploration of high-dimensional parameter spaces. "
            "BayeSED3 is compatible with Linux and macOS (x86_64 and ARM via Rosetta 2), with Windows support through WSL. "
            "Its modular design allows for easy integration of new models and analysis techniques."
        )
        ttk.Label(content_frame, text="Technical Details", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10, 5))
        ttk.Label(content_frame, text=tech_details, wraplength=1160, justify="left").pack(pady=(0, 10))

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Usage section
        usage_info = ("BayeSED3 can be used through this graphical interface, via python script or command-line. "
                      "For detailed usage instructions and examples, please refer to the README file and the example scripts provided with the software package. "
                      "The GUI provides an intuitive way to set up complex SED analysis scenarios with meaningful defaults, while the CLI and python script interfaces are more flexible and allows for batch processing and integration into larger workflows.")
        ttk.Label(content_frame, text="Usage", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10, 5))
        ttk.Label(content_frame, text=usage_info, wraplength=1160, justify="left").pack(pady=(0, 10))

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Website button
        def open_website():
            webbrowser.open_new("https://github.com/hanyk/BayeSED3")

        website_button = ttk.Button(content_frame, text="Website: https://github.com/hanyk/BayeSED3", command=open_website)
        website_button.pack(pady=10)

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Citation section
        citation_info = ("If you use BayeSED3 in your research, please cite:\n"
                         "Han, Y., & Han, Z. 2012, ApJ, 749, 123\n"
                         "Han, Y., & Han, Z. 2014, ApJS, 215, 2\n"
                         "Han, Y., & Han, Z. 2019, ApJS, 240, 3\n"
                         "Han, Y., Fan, L., Zheng, X. Z., Bai, J.-M., & Han, Z. 2023, ApJS, 269, 39\n"
                         "Han, Y., et al. 2024a, in prep.")
        ttk.Label(content_frame, text="Citation", font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(10, 5))
        ttk.Label(content_frame, text=citation_info, wraplength=1160, justify="left").pack(pady=(0, 10))

        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=10)

        # Copyright section
        copyright_info = "© 2012-2024 BayeSED3 Team. All rights reserved."
        ttk.Label(content_frame, text=copyright_info, font=("Helvetica", 14)).pack(pady=10)

        about_window.transient(self.master)
        about_window.grab_set()
        self.master.wait_window(about_window)

if __name__ == "__main__":
    root = tk.Tk()
    gui = BayeSEDGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.stop_execution(), root.destroy()))
    root.mainloop()

