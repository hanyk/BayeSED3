import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import subprocess
import threading
import queue
import sys
from PIL import Image, ImageDraw, ImageTk, ImageFont
import pyperclip
import json
import os
import psutil
import signal
import traceback
from datetime import datetime
from bayesed import BayeSEDInterface, BayeSEDParams, SSPParams, SFHParams, DALParams, MultiNestParams, SysErrParams

class BayeSEDGUI:
    def __init__(self, master):
        self.master = master
        master.title("BayeSED GUI")
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

        # Configure style for larger checkbuttons
        style = ttk.Style()
        style.configure('Large.TCheckbutton', font=('Helvetica', 10))
        
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
        for i, label in enumerate(["Model", "Obs"]):
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

        # Add tooltips for SFH parameters
        sfh_tooltips = {
            "id": "Unique ID for the SFH component",
            "itype_sfh": "SFH type (0-9)",
            "itruncated": "Truncation flag (0: No, 1: Yes)",
            "itype_ceh": "Chemical evolution history type",
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

        # DAL settings
        dal_frame = ttk.Frame(instance_frame)
        dal_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(dal_frame, text="DAL:").grid(row=0, column=0, sticky=tk.W)
        
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
            ttk.Label(dal_frame, text=f"{param}:").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(dal_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
            else:
                widget = ttk.Entry(dal_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=2*i+2, padx=2)
            dal_widgets.append(widget)
            if param == 'id':
                dal_id_widget = widget
                dal_id_widget.config(state='readonly')

        # DEM settings
        dem_frame = ttk.Frame(instance_frame)
        dem_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(dem_frame, text="DEM:").pack(side=tk.LEFT, padx=(0, 5))

        dem_params = [
            ("id", str(new_dem_id), 5),  # DEM ID is always one more than the main ID
            ("imodel", "0", 5, ["0: Greybody", "1: Blackbody", "2: FANN", "3: AKNN"]),
            ("iscalable", "-2", 5),
            ("name", "", 15),
        ]

        dem_widgets = []
        for param_info in dem_params:
            param, default, width = param_info[:3]
            param_frame = ttk.Frame(dem_frame)
            param_frame.pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(param_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
                if param == "imodel":
                    widget.bind("<<ComboboxSelected>>", lambda event, f=instance_frame: self.update_dem_params(event, f))
            else:
                widget = ttk.Entry(param_frame, width=width)
                widget.insert(0, default)
            widget.pack(side=tk.LEFT)
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
            model_frame = ttk.Frame(dem_frame)
            model_widgets = []
            for param, default, width in params:
                param_frame = ttk.Frame(model_frame)
                param_frame.pack(side=tk.LEFT, padx=(0, 5))
                ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT)
                widget = ttk.Entry(param_frame, width=width)
                widget.insert(0, default)
                widget.pack(side=tk.LEFT)
                model_widgets.append(widget)
            self.additional_dem_widgets[model] = (model_frame, model_widgets)

        # Show the initial model's widgets (Greybody by default)
        self.additional_dem_widgets["0"][0].pack(side=tk.LEFT)

        # KIN settings
        kin_frame = ttk.Frame(instance_frame)
        kin_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(kin_frame, text="KIN:").grid(row=0, column=0, sticky=tk.W)
        
        kin_params = [
            ("id", 5),
            ("velscale", 5),
            ("gh_cont", 5),
            ("gh_emis", 5)
        ]
        
        kin_widgets = {}
        for i, (param, width) in enumerate(kin_params):
            ttk.Label(kin_frame, text=f"{param}:").grid(row=0, column=i*2+1, sticky=tk.W, padx=2)
            widget = ttk.Entry(kin_frame, width=width)
            widget.grid(row=0, column=i*2+2, sticky=tk.W, padx=2)
            kin_widgets[param] = widget

        # Set default values and link ID
        kin_widgets['id'].insert(0, ssp_id_widget.get())  # Use the same ID as SSP
        kin_widgets['id'].config(state='readonly')
        kin_widgets['velscale'].insert(0, "10")
        kin_widgets['gh_cont'].insert(0, "0")
        kin_widgets['gh_emis'].insert(0, "0")

        # Add tooltips for KIN parameters
        kin_tooltips = {
            "id": "ID of the model (same as SSP, SFH, DAL)",
            "velscale": "Velocity scale (km/s)",
            "gh_cont": "Number of Gauss-Hermite terms for continuum",
            "gh_emis": "Number of Gauss-Hermite terms for emission lines"
        }
        for param, tooltip in kin_tooltips.items():
            CreateToolTip(kin_widgets[param], tooltip)

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
            'kin_id': kin_widgets['id']
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
            new_id = new_id + 3
            new_igroup = new_igroup + 4
        
        instance_frame = ttk.LabelFrame(self.agn_instances_frame, text=f"AGN {len(self.agn_instances)}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create a dictionary to store the BooleanVars for each component
        component_vars = {
            'main_agn': tk.BooleanVar(value=False),
            'bbb': tk.BooleanVar(value=False),
            'blr': tk.BooleanVar(value=False),
            'feii': tk.BooleanVar(value=False),
            'nlr': tk.BooleanVar(value=False)
        }

        # Main AGN component
        main_agn_frame = ttk.Frame(instance_frame)
        main_agn_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(main_agn_frame, text="Main", variable=component_vars['main_agn'], 
                        command=lambda: self.toggle_component(agn_params_frame, component_vars['main_agn'].get())).grid(row=0, column=0, sticky='w')

        agn_params_frame = ttk.Frame(main_agn_frame)
        agn_params_frame.grid(row=0, column=1, sticky='ew')

        # Initialize AGN parameters
        agn_igroup = ttk.Entry(agn_params_frame, width=8)
        agn_id = ttk.Entry(agn_params_frame, width=8)
        agn_name = ttk.Entry(agn_params_frame, width=12)
        agn_scalable = ttk.Combobox(agn_params_frame, values=["0", "1"], width=5)
        agn_imodel = ttk.Combobox(agn_params_frame, values=["0 (qsosed)", "1 (agnsed)", "2 (fagnsed)", "3 (relagn)", "4 (relqso)", "5 (agnslim)"], width=12)
        agn_icloudy = ttk.Combobox(agn_params_frame, values=["0", "1"], width=5)
        agn_suffix = ttk.Entry(agn_params_frame, width=12)
        agn_w_min = ttk.Entry(agn_params_frame, width=8)
        agn_w_max = ttk.Entry(agn_params_frame, width=8)
        agn_nw = ttk.Entry(agn_params_frame, width=5)

        # First row parameters: igroup, id, name, iscalable, imodel
        ttk.Label(agn_params_frame, text="igroup:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        agn_igroup.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        agn_igroup.insert(0, str(new_igroup))
        CreateToolTip(agn_igroup, "Group ID for the AGN component")

        ttk.Label(agn_params_frame, text="id:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        agn_id.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        agn_id.insert(0, str(new_id))
        CreateToolTip(agn_id, "Unique ID for the AGN component")

        ttk.Label(agn_params_frame, text="name:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        agn_name.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        agn_name.insert(0, "AGN")
        CreateToolTip(agn_name, "Name of the AGN component")

        ttk.Label(agn_params_frame, text="iscalable:").grid(row=0, column=6, sticky=tk.W, padx=5, pady=2)
        agn_scalable.grid(row=0, column=7, sticky=tk.W, padx=5, pady=2)
        agn_scalable.set("1")
        CreateToolTip(agn_scalable, "Whether the component is scalable (0: No, 1: Yes)")

        ttk.Label(agn_params_frame, text="imodel:").grid(row=0, column=8, sticky=tk.W, padx=5, pady=2)
        agn_imodel.grid(row=0, column=9, sticky=tk.W, padx=5, pady=2)
        agn_imodel.set("0 (qsosed)")
        CreateToolTip(agn_imodel, "AGN model type (0: qsosed, 1: agnsed, 2: fagnsed, 3: relagn, 4: relqso, 5: agnslim)")

        # Second row parameters: icloudy, suffix, w_min, w_max, Nw
        ttk.Label(agn_params_frame, text="icloudy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        agn_icloudy.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        agn_icloudy.set("0")
        CreateToolTip(agn_icloudy, "Whether to use Cloudy model (0: No, 1: Yes)")

        ttk.Label(agn_params_frame, text="suffix:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        agn_suffix.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        agn_suffix.insert(0, "disk")  # Set suffix to empty by default
        CreateToolTip(agn_suffix, "Suffix for the AGN component name")

        ttk.Label(agn_params_frame, text="w_min:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        agn_w_min.grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
        agn_w_min.insert(0, "300.0")
        CreateToolTip(agn_w_min, "Minimum wavelength (in microns)")

        ttk.Label(agn_params_frame, text="w_max:").grid(row=1, column=6, sticky=tk.W, padx=5, pady=2)
        agn_w_max.grid(row=1, column=7, sticky=tk.W, padx=5, pady=2)
        agn_w_max.insert(0, "1000.0")
        CreateToolTip(agn_w_max, "Maximum wavelength (in microns)")

        ttk.Label(agn_params_frame, text="Nw:").grid(row=1, column=8, sticky=tk.W, padx=5, pady=2)
        agn_nw.grid(row=1, column=9, sticky=tk.W, padx=5, pady=2)
        agn_nw.insert(0, "200")
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
            ("igroup", 5), ("id", 5), ("name", 10), ("iscalable", 5),
            ("k", 5), ("f_run", 5), ("eps", 5), ("iRad", 5),
            ("iprep", 5), ("Nstep", 5), ("alpha", 5)
        ]

        feii_widgets = {}
        for i, (param, width) in enumerate(aknn_params):
            ttk.Label(feii_content_frame, text=f"{param}:").grid(row=0, column=i*2, sticky=tk.W, padx=2)
            widget = ttk.Entry(feii_content_frame, width=width)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=2)
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

        # Kinematic settings for FeII
        kin_frame = ttk.Frame(feii_content_frame)
        kin_frame.grid(row=1, column=0, columnspan=len(aknn_params)*2, sticky=tk.W, pady=5)

        ttk.Label(kin_frame, text="Kinematics:").pack(side=tk.LEFT, padx=(0, 5))
        kin_params = [("velscale", 5), ("gh_cont", 5), ("gh_emis", 5)]
        kin_widgets = {}
        for param, width in kin_params:
            ttk.Label(kin_frame, text=f"{param}:").pack(side=tk.LEFT, padx=(0, 2))
            widget = ttk.Entry(kin_frame, width=width)
            widget.pack(side=tk.LEFT, padx=(0, 5))
            kin_widgets[param] = widget

        kin_widgets['velscale'].insert(0, "10")
        kin_widgets['gh_cont'].insert(0, "2")
        kin_widgets['gh_emis'].insert(0, "0")

        # Add tooltips for FeII kinematic parameters
        kin_tooltips = {
            "velscale": "Velocity scale for FeII (km/s)",
            "gh_cont": "Number of Gauss-Hermite terms for continuum",
            "gh_emis": "Number of Gauss-Hermite terms for emission"
        }
        for param, tooltip in kin_tooltips.items():
            CreateToolTip(kin_widgets[param], tooltip)

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

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda: self.delete_AGN_instance(instance_frame))
        delete_button.grid(row=5, column=0, sticky='e', padx=5, pady=5)

        # Update the instance dictionary
        self.agn_instances.append({
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
            'nlr_widgets': nlr_widgets
        })

        # Initialize the component visibilities
        for component, var in component_vars.items():
            if component == 'main_agn':
                self.toggle_component(agn_params_frame, var.get())
            else:
                self.toggle_component(locals()[f"{component}_content_frame"], var.get())

        # Configure column weights to allow expansion
        instance_frame.grid_columnconfigure(0, weight=1)
        main_agn_frame.grid_columnconfigure(1, weight=1)
        bbb_frame.grid_columnconfigure(1, weight=1)
        blr_frame.grid_columnconfigure(1, weight=1)
        feii_frame.grid_columnconfigure(1, weight=1)
        nlr_frame.grid_columnconfigure(1, weight=1)

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
            if state:
                child.grid()
            else:
                child.grid_remove()

    # Update the get_agn_settings method to include the use_* flags
    def get_agn_settings(self):
        return [
            {key: (widget.get() if isinstance(widget, (ttk.Entry, ttk.Combobox)) else 
                   widget.get() if isinstance(widget, tk.BooleanVar) else 
                   {k: v.get() for k, v in widget.items()} if isinstance(widget, dict) else None)
             for key, widget in instance.items() if key not in ['frame', 'bbb_frame', 'blr_frame', 'nlr_frame', 'feii_frame']}
            for instance in self.agn_instances
        ]

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)

    def generate_command(self):
        # Start with just the Python interpreter
        command = []
        
        # Get the number of MPI processes if specified
        np = self.mpi_processes.get().strip()
        if np:
            command.extend(["--np", np])
        
        # Basic settings
        input_type = self.input_type.get().split()[0]
        command.extend(["-i", f"{input_type},{self.input_file.get()}"])
        command.extend(["--outdir", self.outdir.get()])
        command.extend(["-v", self.verbose.get()])
        
        # Output settings (now part of Basic Settings)
        if self.save_bestfit.get():
            save_type = self.save_bestfit_type.get().split()[0]
            command.extend(["--save_bestfit", save_type])
        if self.save_sample_par.get():
            command.append("--save_sample_par")
        if self.save_sample_obs.get():
            command.append("--save_sample_obs")
        
        # Add new save parameters to the command
        if self.save_pos_sfh.get():
            ngrid = self.save_pos_sfh_ngrid.get()
            ilog = self.save_pos_sfh_ilog.get()
            command.extend(["--save_pos_sfh", f"{ngrid},{ilog}"])
        if self.save_pos_spec.get():
            command.append("--save_pos_spec")
        if self.save_sample_spec.get():
            command.append("--save_sample_spec")
        if self.save_summary.get():
            command.append("--save_summary")
        
        # Galaxy instance settings
        for instance in self.galaxy_instances:
            ssp_values = [widget.get() for widget in instance['ssp']]
            sfh_values = [widget.get() for widget in instance['sfh']]
            dal_values = [widget.get() for widget in instance['dal']]
            dem_values = [widget.get() for widget in instance['dem']]
            kin_values = [widget.get() for widget in instance['kin'].values()]
            
            if all(ssp_values):
                command.extend(["-ssp", ",".join(ssp_values)])
                
            if all(sfh_values):
                command.extend(["--sfh", ",".join(sfh_values[:4])])  # id, itype_sfh, itruncated, itype_ceh
                if int(sfh_values[1]) == 9:  # If itype_sfh is 9 (Nonparametric)
                    np_sfh_values = sfh_values[4:]  # np_prior_type, np_interp_method, np_num_bins, np_regul
                    command.extend(["--np_sfh", ",".join(np_sfh_values)])
                
            if all(dal_values):
                command.extend(["--dal", ",".join(dal_values)])
            
            if all(dem_values):
                imodel = dem_values[1]  # The imodel value
                igroup = ssp_values[0]  # Use SSP's igroup for DEM
                additional_values = []
                for widget in self.additional_dem_widgets[imodel][1]:
                    try:
                        additional_values.append(widget.get())
                    except tk.TclError:
                        print(f"Warning: Widget for DEM model {imodel} no longer exists.")
                all_dem_values = dem_values + additional_values
                
                try:
                    if imodel == "0":  # Greybody
                        if len(all_dem_values) >= 8:
                            command.extend(["--greybody", f"{igroup},{all_dem_values[0]},{all_dem_values[2]},{all_dem_values[3]},{all_dem_values[4]},{all_dem_values[5]},{all_dem_values[6]},{all_dem_values[7]}"])
                        else:
                            print(f"Warning: Not enough values for Greybody model. Expected 8, got {len(all_dem_values)}.")
                    elif imodel == "1":  # Blackbody
                        if len(all_dem_values) >= 7:
                            command.extend(["--blackbody", f"{igroup},{all_dem_values[0]},{all_dem_values[2]},{all_dem_values[3]},{all_dem_values[5]},{all_dem_values[6]},{all_dem_values[7]}"])
                        else:
                            print(f"Warning: Not enough values for Blackbody model. Expected 7, got {len(all_dem_values)}.")
                    elif imodel == "2":  # FANN
                        if len(all_dem_values) >= 4:
                            command.extend(["-a", f"{igroup},{all_dem_values[0]},{all_dem_values[2]},{all_dem_values[3]}"])
                        else:
                            print(f"Warning: Not enough values for FANN model. Expected 4, got {len(all_dem_values)}.")
                    elif imodel == "3":  # AKNN
                        if len(all_dem_values) >= 11:
                            command.extend(["-k", f"{igroup},{all_dem_values[0]},{all_dem_values[2]},{all_dem_values[3]},{all_dem_values[4]},{all_dem_values[5]},{all_dem_values[6]},{all_dem_values[7]},{all_dem_values[8]},{all_dem_values[9]},{all_dem_values[10]}"])
                        else:
                            print(f"Warning: Not enough values for AKNN model. Expected 11, got {len(all_dem_values)}.")
                except (KeyError, IndexError) as e:
                    print(f"Error processing DEM model {imodel}: {str(e)}")

            if all(kin_values):
                command.extend(['--kin', ','.join(kin_values)])

        # AGN settings
        for agn in self.agn_instances:
            if agn['component_vars']['main_agn'].get():
                agn_igroup = agn['agn_igroup'].get()
                agn_id = agn['agn_id'].get()
                agn_name = agn['name'].get()
                agn_scalable = agn['iscalable'].get()
                agn_imodel = agn['imodel'].get().split()[0]  # Extract the number from "x (desc)"
                agn_icloudy = agn['icloudy'].get()
                agn_suffix = agn['suffix'].get()
                agn_w_min = agn['w_min'].get()
                agn_w_max = agn['w_max'].get()
                agn_nw = agn['nw'].get()
        
                if agn_igroup and agn_id and agn_name and agn_scalable and agn_imodel and agn_icloudy and agn_suffix and agn_w_min and agn_w_max and agn_nw:
                    command.extend([
                        "--AGN",
                        f"{agn_igroup},{agn_id},{agn_name},{agn_scalable},{agn_imodel},{agn_icloudy},{agn_suffix},{agn_w_min},{agn_w_max},{agn_nw}"
                    ])

            if agn['component_vars']['bbb'].get():
                bbb_id = int(agn['agn_id'].get()) + 1
                command.extend([
                    "-bbb",
                    f"{bbb_id},{bbb_id},{agn['bbb_name'].get()},1,{agn['bbb_w_min'].get()},{agn['bbb_w_max'].get()},{agn['bbb_nw'].get()}"
                ])

            if agn['component_vars']['blr'].get():
                blr_params = [agn['blr_widgets'][p].get() for p in ['igroup', 'id', 'name', 'iscalable', 'file', 'R', 'Nsample', 'Nkin']]
                command.extend(["-ls1", ",".join(blr_params)])

            if agn['component_vars']['feii'].get():
                feii_params = [agn['feii_widgets'][p].get() for p in ['igroup', 'id', 'name', 'iscalable', 'k', 'f_run', 'eps', 'iRad', 'iprep', 'Nstep', 'alpha']]
                command.extend(["-k", ",".join(feii_params)])
                kin_params = [agn['feii_widgets']['id'].get()] + [agn['kin_widgets'][p].get() for p in ['velscale', 'gh_cont', 'gh_emis']]
                command.extend(["--kin", ",".join(kin_params)])

            if agn['component_vars']['nlr'].get():
                nlr_params = [agn['nlr_widgets'][p].get() for p in ['igroup', 'id', 'name', 'iscalable', 'file', 'R', 'Nsample', 'Nkin']]
                command.extend(["-ls1", ",".join(nlr_params)])
        
        # Advanced settings
        if self.use_multinest.get():
            multinest_values = [widget.get() for widget in self.multinest_widgets.values()]
            command.extend(["--multinest", ",".join(multinest_values)])
        
        if self.use_nnlm.get():
            nnlm_values = [widget.get() for widget in self.nnlm_widgets.values()]
            if all(nnlm_values):
                command.extend(["--NNLM", ",".join(nnlm_values)])

        if self.use_ndumper.get():
            ndumper_values = [widget.get() for widget in self.ndumper_widgets.values()]
            if all(ndumper_values):
                command.extend(["--Ndumper", ",".join(ndumper_values)])

        if self.use_gsl.get():
            gsl_integration_values = [self.gsl_widgets[p].get() for p in ["integration_epsabs", "integration_epsrel", "integration_limit"]]
            if all(gsl_integration_values):
                command.extend(["--gsl_integration_qag", ",".join(gsl_integration_values)])
            
            gsl_multifit_values = [self.gsl_widgets[p].get() for p in ["multifit_type", "multifit_tune"]]
            if all(gsl_multifit_values):
                command.extend(["--gsl_multifit_robust", ",".join(gsl_multifit_values)])

        if self.use_misc.get():
            for param, widget in self.misc_widgets.items():
                value = widget.get()
                if value:
                    command.extend([f"--{param}", value])

        # Add output options to the command
        if self.output_mock_photometry.get():
            output_type = self.output_mock_photometry_type.get().split()[0]
            command.extend(["--output_mock_photometry", output_type])
        if self.output_mock_spectra.get():
            command.append("--output_mock_spectra")
        if self.output_model_absolute_magnitude.get():
            command.append("--output_model_absolute_magnitude")
        if self.output_pos_obs.get():
            command.append("--output_pos_obs")
        if self.suffix.get():
            command.extend(["--suffix", self.suffix.get()])

        # Add cosmology parameters
        if self.use_cosmology.get():
            h0 = self.cosmology_params['H0'].get()
            omigaA = self.cosmology_params['omigaA'].get()
            omigam = self.cosmology_params['omigam'].get()
            command.extend(["--cosmology", f"{h0},{omigaA},{omigam}"])

        # Add IGM model
        if self.use_igm.get():
            command.extend(["--IGM", self.igm_model.get()])

        # Add redshift parameters if enabled
        if self.use_redshift.get():
            redshift_values = [widget.get() for widget in self.redshift_widgets]
            if all(redshift_values):
                command.extend(["--z", ",".join(redshift_values)])

        # Boolean options
        if self.no_photometry_fit.get():
            command.append("--no_photometry_fit")
        if self.no_spectra_fit.get():
            command.append("--no_spectra_fit")
        if self.unweighted_samples.get():
            command.append("--unweighted_samples")

        # Add new parameters
        if self.filters.get():
            command.extend(["--filters", self.filters.get()])
        if self.filters_selected.get():
            command.extend(["--filters_selected", self.filters_selected.get()])
        
        if self.use_sfr.get():
            command.extend(["--SFR_over", self.sfr_myr_entry.get()])
        
        if self.use_snr.get():
            command.extend(["--SNRmin1", self.snrmin1.get()])
            command.extend(["--SNRmin2", self.snrmin2.get()])
        
        if self.use_build_sedlib.get():
            command.extend(["--build_sedlib", self.build_sedlib.get().split()[0]])
        
        if self.priors_only.get():
            command.append("--priors_only")

        # Output SFH
        if self.use_output_sfh.get():
            command.extend(["--output_SFH", f"{self.output_sfh_ntimes.get()},{self.output_sfh_ilog.get()}"])

        # Systematic Error
        if self.use_sys_err.get():
            mod_values = ",".join([widget.get() for widget in self.sys_err_widgets[:5]])
            obs_values = ",".join([widget.get() for widget in self.sys_err_widgets[5:]])
            command.extend(["--sys_err_mod", mod_values])
            command.extend(["--sys_err_obs", obs_values])

        # Ntest
        ntest = self.ntest.get().strip()
        if ntest:
            command.extend(["--Ntest", ntest])

        return command

    def run_bayesed(self):
        if self.run_button['text'] == "Run":
            command = self.generate_command()
            
            self.update_output("Executing command: " + " ".join(command) + "\n")
            
            np = self.mpi_processes.get().strip()
            ntest = self.ntest.get().strip()
            
            self.output_queue = queue.Queue()
            self.stop_output_thread = threading.Event()
            threading.Thread(target=self.execute_command, args=(command, np, ntest), daemon=True).start()
            self.master.after(100, self.check_output_queue)

            # Change button text to "Stop"
            self.run_button.config(text="Stop", command=self.stop_bayesed)
        else:
            self.stop_bayesed()

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
        try:
            # Use subprocess.STARTUPINFO to hide console window on Windows
            startupinfo = None
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            bayesed = BayeSEDInterface(mpi_mode='1', np=int(np) if np else None, Ntest=int(ntest) if ntest else None)
            bayesed.run(command)
            
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
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.NORMAL, font=self.standard_font)  # Apply standard font to output text
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

        # Hide all additional parameter frames
        for model_frame, _ in self.additional_dem_widgets.values():
            model_frame.pack_forget()

        # Show the frame for the selected model
        if imodel in self.additional_dem_widgets:
            self.additional_dem_widgets[imodel][0].pack(side=tk.LEFT)

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
                f.write("from bayesed import (BayeSEDInterface, BayeSEDParams, SSPParams, SFHParams, DALParams,\n")
                f.write("                     MultiNestParams, SysErrParams, AGNParams, BigBlueBumpParams, LineParams,\n")
                f.write("                     AKNNParams, KinParams, CosmologyParams, ZParams, NNLMParams, NdumperParams,\n")
                f.write("                     GSLIntegrationQAGParams, GSLMultifitRobustParams)\n\n")

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
                
                f.write("    params = BayeSEDParams(\n")
                
                # Basic settings
                f.write(f"        input_type={self.input_type.get().split()[0]},\n")
                f.write(f"        input_file='{self.input_file.get()}',\n")
                f.write(f"        outdir='{self.outdir.get()}',\n")
                f.write(f"        verbose={self.verbose.get()},\n")
                
                # Output settings
                if self.save_bestfit.get():
                    f.write(f"        save_bestfit={self.save_bestfit_type.get().split()[0]},\n")
                if self.save_sample_par.get():
                    f.write("        save_sample_par=True,\n")
                if self.save_sample_obs.get():
                    f.write("        save_sample_obs=True,\n")
                if self.save_pos_sfh.get():
                    f.write(f"        save_pos_sfh='{self.save_pos_sfh_ngrid.get()},{self.save_pos_sfh_ilog.get()}',\n")
                if self.save_pos_spec.get():
                    f.write("        save_pos_spec=True,\n")
                if self.save_sample_spec.get():
                    f.write("        save_sample_spec=True,\n")
                if self.save_summary.get():
                    f.write("        save_summary=True,\n")
                
                # Galaxy instance settings
                if self.galaxy_instances:
                    f.write("        ssp=[")
                    for instance in self.galaxy_instances:
                        ssp_values = [widget.get() for widget in instance['ssp']]
                        f.write(f"SSPParams(igroup={ssp_values[0]}, id={ssp_values[1]}, name='{ssp_values[2]}', iscalable={ssp_values[3]}, k={ssp_values[4]}, f_run={ssp_values[5]}, Nstep={ssp_values[6]}, i0={ssp_values[7]}, i1={ssp_values[8]}, i2={ssp_values[9]}, i3={ssp_values[10]}),")
                    f.write("],\n")
                    
                    f.write("        sfh=[")
                    for instance in self.galaxy_instances:
                        sfh_values = [widget.get() for widget in instance['sfh']]
                        f.write(f"SFHParams(id={sfh_values[0]}, itype_sfh={sfh_values[1]}, itruncated={sfh_values[2]}, itype_ceh={sfh_values[3]}),")
                    f.write("],\n")
                    
                    f.write("        dal=[")
                    for instance in self.galaxy_instances:
                        dal_values = [widget.get() for widget in instance['dal']]
                        f.write(f"DALParams(id={dal_values[0]}, con_eml_tot={dal_values[1]}, ilaw={dal_values[2]}),")
                    f.write("],\n")
                    
                    f.write("        kin=[")
                    for instance in self.galaxy_instances:
                        kin_values = [widget.get() for widget in instance['kin'].values()]
                        f.write(f"KinParams(id={kin_values[0]}, velscale={kin_values[1]}, num_gauss_hermites_continuum={kin_values[2]}, num_gauss_hermites_emission={kin_values[3]}),")
                    f.write("],\n")
                
                if self.use_sys_err.get():
                    sys_err_mod_values = [widget.get() for widget in self.sys_err_widgets[:5]]
                    sys_err_obs_values = [widget.get() for widget in self.sys_err_widgets[5:]]
                    f.write(f"        sys_err_mod=SysErrParams(iprior_type={sys_err_mod_values[0]}, is_age={sys_err_mod_values[1]}, min={sys_err_mod_values[2]}, max={sys_err_mod_values[3]}, nbin={sys_err_mod_values[4]}),\n")
                    f.write(f"        sys_err_obs=SysErrParams(iprior_type={sys_err_obs_values[0]}, is_age={sys_err_obs_values[1]}, min={sys_err_obs_values[2]}, max={sys_err_obs_values[3]}, nbin={sys_err_obs_values[4]}),\n")
                
                # Other settings
                if self.filters.get():
                    f.write(f"        filters='{self.filters.get()}',\n")
                if self.filters_selected.get():
                    f.write(f"        filters_selected='{self.filters_selected.get()}',\n")
                if self.no_photometry_fit.get():
                    f.write("        no_photometry_fit=True,\n")
                if self.no_spectra_fit.get():
                    f.write("        no_spectra_fit=True,\n")
                if self.unweighted_samples.get():
                    f.write("        unweighted_samples=True,\n")
                if self.priors_only.get():
                    f.write("        priors_only=True,\n")
                f.write("    )\n\n")  # Close the BayeSEDParams initialization
                
                # AGN settings
                agn_settings = self.get_agn_settings()
                if agn_settings:
                    f.write("    # AGN components\n")
                    for agn in agn_settings:
                        if agn['component_vars']['main_agn']:
                            f.write(f"    params.AGN = [AGNParams(igroup={agn['agn_igroup']}, id={agn['agn_id']}, name='{agn['name']}', iscalable={agn['iscalable']}, imodel={agn['imodel'].split()[0]}, icloudy={agn['icloudy']}, suffix='{agn['suffix']}', w_min={agn['w_min']}, w_max={agn['w_max']}, Nw={agn['nw']})]\n")
                        
                        if agn['component_vars']['bbb']:
                            f.write(f"    params.big_blue_bump = [BigBlueBumpParams(igroup={agn['bbb_igroup']}, id={agn['bbb_id']}, name='{agn['bbb_name']}', iscalable=1, w_min={agn['bbb_w_min']}, w_max={agn['bbb_w_max']}, Nw={agn['bbb_nw']})]\n")
                        
                        if agn['component_vars']['blr']:
                            blr = agn['blr_widgets']
                            f.write(f"    params.lines1 = [LineParams(igroup={blr['igroup']}, id={blr['id']}, name='{blr['name']}', iscalable={blr['iscalable']}, file='{blr['file']}', R={blr['R']}, Nsample={blr['Nsample']}, Nkin={blr['Nkin']})]\n")
                        
                        if agn['component_vars']['feii']:
                            feii = agn['feii_widgets']
                            f.write(f"    params.aknn = [AKNNParams(igroup={feii['igroup']}, id={feii['id']}, name='{feii['name']}', iscalable={feii['iscalable']}, k={feii['k']}, f_run={feii['f_run']}, eps={feii['eps']}, iRad={feii['iRad']}, iprep={feii['iprep']}, Nstep={feii['Nstep']}, alpha={feii['alpha']})]\n")
                            kin = agn['kin_widgets']
                            f.write(f"    params.kin.append(KinParams(id={feii['id']}, velscale={kin['velscale']}, num_gauss_hermites_continuum={kin['gh_cont']}, num_gauss_hermites_emission={kin['gh_emis']}))\n")
                        
                        if agn['component_vars']['nlr']:
                            nlr = agn['nlr_widgets']
                            f.write(f"    params.lines1.append(LineParams(igroup={nlr['igroup']}, id={nlr['id']}, name='{nlr['name']}', iscalable={nlr['iscalable']}, file='{nlr['file']}', R={nlr['R']}, Nsample={nlr['Nsample']}, Nkin={nlr['Nkin']}))\n")

                # Cosmology settings
                f.write("\n\n    # Cosmology settings\n")
                if self.use_cosmology.get():
                    f.write(f"    params.cosmology = CosmologyParams(\n")
                    f.write(f"        H0={self.cosmology_params['H0'].get()},\n")
                    f.write(f"        omigaA={self.cosmology_params['omigaA'].get()},\n")
                    f.write(f"        omigam={self.cosmology_params['omigam'].get()}\n")
                    f.write(f"    )\n\n")

                # IGM model
                f.write("    # IGM model\n")
                if self.use_igm.get():
                    f.write(f"    params.IGM = {self.igm_model.get()}\n\n")

                # Redshift parameters
                f.write("    # Redshift parameters\n")
                if self.use_redshift.get():
                    f.write(f"    params.z = ZParams(\n")
                    f.write(f"        iprior_type={self.redshift_params['iprior_type'].get()},\n")
                    f.write(f"        is_age={self.redshift_params['is_age'].get()},\n")
                    f.write(f"        min={self.redshift_params['min'].get()},\n")
                    f.write(f"        max={self.redshift_params['max'].get()},\n")
                    f.write(f"        nbin={self.redshift_params['nbin'].get()}\n")
                    f.write(f"    )\n\n")

                # Advanced settings
                if self.use_multinest.get():
                    f.write("    params.multinest = MultiNestParams(\n")
                    for i, (key, widget) in enumerate(self.multinest_widgets.items()):
                        f.write(f"        {key}={widget.get()}")
                        if i < len(self.multinest_widgets) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("    )\n\n")

                if self.use_nnlm.get():
                    f.write("    params.NNLM = NNLMParams(\n")
                    for i, (key, widget) in enumerate(self.nnlm_widgets.items()):
                        f.write(f"        {key}={widget.get()}")
                        if i < len(self.nnlm_widgets) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("    )\n\n")

                if self.use_ndumper.get():
                    f.write("    params.Ndumper = NdumperParams(\n")
                    for i, (key, widget) in enumerate(self.ndumper_widgets.items()):
                        f.write(f"        {key}={widget.get()}")
                        if i < len(self.ndumper_widgets) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("    )\n\n")

                if self.use_gsl.get():
                    f.write("    params.gsl_integration_qag = GSLIntegrationQAGParams(\n")
                    gsl_int_params = ["epsabs", "epsrel", "limit"]
                    for i, key in enumerate(gsl_int_params):
                        f.write(f"        {key}={self.gsl_widgets[f'integration_{key}'].get()}")
                        if i < len(gsl_int_params) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("    )\n")
                    f.write("    params.gsl_multifit_robust = GSLMultifitRobustParams(\n")
                    f.write(f"        type='{self.gsl_widgets['multifit_type'].get()}',\n")
                    f.write(f"        tune={self.gsl_widgets['multifit_tune'].get()}\n")
                    f.write("    )\n\n")

                if self.use_misc.get():
                    for key, widget in self.misc_widgets.items():
                        value = widget.get()
                        if value:
                            if key == 'cl':
                                f.write(f"    params.cl = '{value}'\n")
                            else:
                                f.write(f"    params.{key} = {value}\n")
                    f.write("\n")

                f.write("    bayesed.run(params)\n")
                f.write("if __name__ == '__main__':\n")
                f.write("    run_bayesed()\n")
            
            messagebox.showinfo("Save Successful", f"Script saved to {filename}")

# Add the following tooltip class if not already present
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
                         font=("Arial", "10", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

if __name__ == "__main__":
    root = tk.Tk()
    gui = BayeSEDGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.stop_execution(), root.destroy()))
    root.mainloop()

