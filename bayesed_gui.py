import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import subprocess
import threading
import queue
import sys
from PIL import Image, ImageDraw, ImageTk, ImageFont
import pyperclip
import json

class BayeSEDGUI:
    def __init__(self, master):
        self.master = master
        master.title("BayeSED GUI")
        master.geometry("1400x800")
        
        # Define a standard font
        self.standard_font = ('Helvetica', 10)
        
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
        CreateToolTip(self.input_file, "Input file containing observed photometric SEDs")

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
        ttk.Label(input_frame, text="Filters:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters = ttk.Entry(input_frame, width=40)
        self.filters.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_filters).grid(row=4, column=2, padx=5, pady=2)
        CreateToolTip(self.filters, "File containing the definition of filters")

        # Filters Selected
        ttk.Label(input_frame, text="Filters Selected:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters_selected = ttk.Entry(input_frame, width=40)
        self.filters_selected.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(input_frame, text="Browse", command=self.browse_filters_selected).grid(row=5, column=2, padx=5, pady=2)
        CreateToolTip(self.filters_selected, "File containing all used filters in the observation and select those needed")

        # Priors Only
        ttk.Checkbutton(input_frame, text="Priors Only", variable=self.priors_only, style='Large.TCheckbutton').grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Test priors by setting the loglike for observational data to be zero")

        # No photometry fit
        ttk.Checkbutton(input_frame, text="No photometry fit", variable=self.no_photometry_fit, style='Large.TCheckbutton').grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Do not fit photometric data even if it is presented")

        # No spectra fit
        ttk.Checkbutton(input_frame, text="No spectra fit", variable=self.no_spectra_fit, style='Large.TCheckbutton').grid(row=8, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(input_frame.winfo_children()[-1], "Do not fit spectra data even if it is presented")

        # SNR Settings
        snr_frame = ttk.Frame(input_frame)
        snr_frame.grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Checkbutton(snr_frame, variable=self.use_snr, 
                        command=lambda: self.toggle_widgets([self.snrmin1, self.snrmin2], self.use_snr.get()),
                        style='Large.TCheckbutton', text="SNR Settings").pack(side=tk.LEFT, padx=5)
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
                        style='Large.TCheckbutton', text="Systematic Error").pack(side=tk.LEFT)
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
                        style='Large.TCheckbutton', text="Build SED Library").pack(side=tk.LEFT)
        CreateToolTip(build_sed_frame.winfo_children()[-1], "Build a SED library using the employed models")

        self.build_sedlib = ttk.Combobox(build_sed_frame, values=["0 (Rest)", "1 (Observed)"], width=15, state="disabled")
        self.build_sedlib.set("0 (Rest)")
        self.build_sedlib.pack(side=tk.LEFT, padx=5)
        CreateToolTip(self.build_sedlib, "0: Rest frame, 1: Observed frame")

        # Use unweighted samples
        ttk.Checkbutton(output_frame, text="Use unweighted samples", variable=self.unweighted_samples, style='Large.TCheckbutton').grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Use unweighted posterior samples")

        # SFR Settings
        sfr_frame = ttk.Frame(output_frame)
        sfr_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(sfr_frame, variable=self.use_sfr, 
                        command=lambda: self.toggle_widgets([self.sfr_myr_entry], self.use_sfr.get()),
                        style='Large.TCheckbutton', text="Output SFR over").pack(side=tk.LEFT)
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
                        style='Large.TCheckbutton', text="Output SFH").pack(side=tk.LEFT)
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

        # Run button
        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_bayesed)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Export and Import buttons
        ttk.Button(control_frame, text="Export Settings", command=self.export_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Import Settings", command=self.import_settings).pack(side=tk.LEFT, padx=5)

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

        # SSP settings
        ssp_frame = ttk.Frame(instance_frame)
        ssp_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ssp_frame, text="SSP:").grid(row=0, column=0, sticky=tk.W)
        
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
            ("itype_ceh", "0", 5, ["0: No CEH", "1: linear mapping model"])
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

        # Create the instance dictionary
        new_instance = {
            'frame': instance_frame,
            'ssp': ssp_widgets,
            'sfh': sfh_widgets,
            'dal': dal_widgets,
            'dem': dem_widgets,
            'ssp_id': ssp_id_widget,
            'sfh_id': sfh_id_widget,
            'dal_id': dal_id_widget,
            'dem_id': dem_id_widget
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
        multinest_frame = ttk.Frame(advanced_frame)
        multinest_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Checkbutton(multinest_frame, variable=self.use_multinest, 
                        command=lambda: self.toggle_widgets(self.multinest_widgets.values(), self.use_multinest.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        multinest_content = ttk.LabelFrame(multinest_frame, text="MultiNest Settings")
        multinest_content.pack(side=tk.LEFT, expand=True, fill=tk.X)

        multinest_params = [
            ("INS", "Importance Nested Sampling flag (0 or 1)"),
            ("mmodal", "Multimodal flag (0 or 1)"),
            ("ceff", "Constant efficiency mode flag (0 or 1)"),
            ("nlive", "Number of live points"),
            ("efr", "Sampling efficiency"),
            ("tol", "Tolerance for termination"),
            ("updInt", "Update interval for posterior output"),
            ("Ztol", "Evidence tolerance"),
            ("seed", "Random seed (0 for system time)"),
            ("fb", "Feedback level (0-3)"),
            ("resume", "Resume from a previous run (0 or 1)"),
            ("outfile", "Write output files (0 or 1)"),
            ("logZero", "Log of Zero (points with loglike < logZero will be ignored)"),
            ("maxiter", "Maximum number of iterations"),
            ("acpt", "Acceptance rate")
        ]

        default_values = "1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01".split(',')
        self.multinest_widgets = {}

        for i, (param, tooltip) in enumerate(multinest_params):
            row = i // 3
            col = i % 3 * 2
            ttk.Label(multinest_content, text=f"{param}:").grid(row=row+1, column=col, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(multinest_content, width=8)
            widget.insert(0, default_values[i] if i < len(default_values) else "")
            widget.grid(row=row+1, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.multinest_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # NNLM Settings
        nnlm_frame = ttk.Frame(advanced_frame)
        nnlm_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Checkbutton(nnlm_frame, variable=self.use_nnlm, 
                        command=lambda: self.toggle_widgets(self.nnlm_widgets.values(), self.use_nnlm.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        nnlm_content = ttk.LabelFrame(nnlm_frame, text="NNLM Settings")
        nnlm_content.pack(side=tk.LEFT, expand=True, fill=tk.X)

        nnlm_params = [
            ("method", "Method (0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl)", "0"),
            ("Niter1", "Number of iterations for first step", "10000"),
            ("tol1", "Tolerance for first step", "0"),
            ("Niter2", "Number of iterations for second step", "10"),
            ("tol2", "Tolerance for second step", "0.01"),
            ("p1", "Parameter p1 for NNLM algorithm", "0.05"),
            ("p2", "Parameter p2 for NNLM algorithm", "0.95")
        ]
        self.nnlm_widgets = {}
        for i, (param, tooltip, default) in enumerate(nnlm_params):
            ttk.Label(nnlm_content, text=f"{param}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(nnlm_content, width=10)
            widget.insert(0, default)
            widget.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            self.nnlm_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Ndumper Settings
        ndumper_frame = ttk.Frame(advanced_frame)
        ndumper_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Checkbutton(ndumper_frame, variable=self.use_ndumper, 
                        command=lambda: self.toggle_widgets(self.ndumper_widgets.values(), self.use_ndumper.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        ndumper_content = ttk.LabelFrame(ndumper_frame, text="Ndumper Settings")
        ndumper_content.pack(side=tk.LEFT, expand=True, fill=tk.X)

        ndumper_params = [
            ("max_number", "Maximum number of samples to dump", "1"),
            ("iconverged_min", "Minimum convergence flag", "0"),
            ("Xmin_squared_Nd", "Xmin^2/Nd value (-1 for no constraint)", "-1")
        ]
        self.ndumper_widgets = {}
        for i, (param, tooltip, default) in enumerate(ndumper_params):
            ttk.Label(ndumper_content, text=f"{param}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(ndumper_content, width=10)
            widget.insert(0, default)
            widget.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            self.ndumper_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # GSL Integration and Multifit Settings
        gsl_frame = ttk.Frame(advanced_frame)
        gsl_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Checkbutton(gsl_frame, variable=self.use_gsl, 
                        command=lambda: self.toggle_widgets(self.gsl_widgets.values(), self.use_gsl.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        gsl_content = ttk.LabelFrame(gsl_frame, text="GSL Settings")
        gsl_content.pack(side=tk.LEFT, expand=True, fill=tk.X)

        gsl_params = [
            ("integration_epsabs", "Absolute error for GSL integration", "0"),
            ("integration_epsrel", "Relative error for GSL integration", "0.1"),
            ("integration_limit", "Limit for GSL integration", "1000"),
            ("multifit_type", "Multifit type (ols or huber)", "ols"),
            ("multifit_tune", "Tuning parameter for robust fitting", "1.0")
        ]
        self.gsl_widgets = {}
        for i, (param, tooltip, default) in enumerate(gsl_params):
            ttk.Label(gsl_content, text=f"{param}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(gsl_content, width=10)
            widget.insert(0, default)
            widget.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            self.gsl_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Other Miscellaneous Settings
        misc_frame = ttk.Frame(advanced_frame)
        misc_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Checkbutton(misc_frame, variable=self.use_misc, 
                        command=lambda: self.toggle_widgets(self.misc_widgets.values(), self.use_misc.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        misc_content = ttk.LabelFrame(misc_frame, text="Other Settings")
        misc_content.pack(side=tk.LEFT, expand=True, fill=tk.X)

        misc_params = [
            ("NfilterPoints", "Number of filter points for interpolation", "30"),
            ("Nsample", "Number of samples for catalog creation or SED library building", ""),
            ("Ntest", "Number of objects for test run", ""),
            ("niteration", "Number of iterations", "0"),
            ("logZero", "Log of Zero (points with loglike < logZero will be ignored)", "-1e90"),
            ("lw_max", "Max line coverage in km/s for emission line model creation", "10000"),
            ("cl", "Confidence levels for output estimates", "0.68,0.95")
        ]
        self.misc_widgets = {}
        for i, (param, tooltip, default) in enumerate(misc_params):
            row = i // 3
            col = i % 3 * 2
            ttk.Label(misc_content, text=f"{param}:").grid(row=row+1, column=col, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(misc_content, width=10)
            widget.insert(0, default)
            widget.grid(row=row+1, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.misc_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Configure grid weights
        advanced_frame.grid_columnconfigure(0, weight=1)
        advanced_frame.grid_columnconfigure(1, weight=1)

        # At the end of create_advanced_tab method, add these lines:
        self.toggle_widgets(self.multinest_widgets.values(), self.use_multinest.get())
        self.toggle_widgets(self.nnlm_widgets.values(), self.use_nnlm.get())
        self.toggle_widgets(self.ndumper_widgets.values(), self.use_ndumper.get())
        self.toggle_widgets(self.gsl_widgets.values(), self.use_gsl.get())
        self.toggle_widgets(self.misc_widgets.values(), self.use_misc.get())

    def toggle_widgets(self, widgets, state):
        for widget in widgets:
            if isinstance(widget, ttk.Radiobutton):
                widget.config(state="normal" if state else "disabled")
            elif hasattr(widget, 'config'):
                if state:
                    widget.config(state="normal")
                    widget.config(foreground="black")
                else:
                    widget.config(state="disabled")
                    widget.config(foreground="grey")
            
            # Handle the label associated with the widget
            parent = widget.master
            for child in parent.winfo_children():
                if isinstance(child, ttk.Label) and child.grid_info()['row'] == widget.grid_info()['row']:
                    if state:
                        child.config(foreground="black")
                    else:
                        child.config(foreground="grey")

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
        main_agn_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(main_agn_frame, text="Main", variable=component_vars['main_agn'], 
                        command=lambda: self.toggle_component(agn_params_frame, component_vars['main_agn'].get())).grid(row=0, column=0, sticky=tk.W)

        agn_params_frame = ttk.Frame(main_agn_frame)
        agn_params_frame.grid(row=0, column=1, sticky=tk.W)

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
        bbb_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(bbb_frame, text="BBB", variable=component_vars['bbb'], 
                        command=lambda: self.toggle_component(bbb_content_frame, component_vars['bbb'].get())).grid(row=0, column=0, sticky=tk.W)

        bbb_content_frame = ttk.Frame(bbb_frame)
        bbb_content_frame.grid(row=0, column=1, sticky=tk.W)

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
        blr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(blr_frame, text="BLR", variable=component_vars['blr'], 
                        command=lambda: self.toggle_component(blr_content_frame, component_vars['blr'].get())).pack(side=tk.LEFT, padx=(0, 5))

        blr_content_frame = ttk.Frame(blr_frame)
        blr_content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        feii_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(feii_frame, text="FeII", variable=component_vars['feii'], 
                        command=lambda: self.toggle_component(feii_content_frame, component_vars['feii'].get())).pack(side=tk.LEFT, padx=(0, 5))

        feii_content_frame = ttk.Frame(feii_frame)
        feii_content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        nlr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(nlr_frame, text="NLR", variable=component_vars['nlr'], 
                        command=lambda: self.toggle_component(nlr_content_frame, component_vars['nlr'].get())).pack(side=tk.LEFT, padx=(0, 5))

        nlr_content_frame = ttk.Frame(nlr_frame)
        nlr_content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        delete_button.pack(side=tk.RIGHT, padx=5, pady=5)

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

    # Add this new method to toggle component visibility
    def toggle_component(self, frame, state):
        for child in frame.winfo_children():
            if child.winfo_class() == 'TCheckbutton':
                continue  # Skip the checkbox itself
            if state:
                if child.winfo_manager() == 'grid':
                    child.grid()
                elif child.winfo_manager() == 'pack':
                    child.pack()
            else:
                if child.winfo_manager() == 'grid':
                    child.grid_remove()
                elif child.winfo_manager() == 'pack':
                    child.pack_forget()

    # Update the generate_command method to include only selected components
    def generate_command(self):
        command = [self.bayesed_path.get()]

        # ... (rest of the code remains unchanged)

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

        # ... (rest of the code remains unchanged)

        return command
        nlr_file.insert(0, "observation/test/lines_NLR.txt")
        ttk.Label(nlr_frame, text="R:").pack(side=tk.LEFT, padx=(0, 5))
        nlr_r = ttk.Entry(nlr_frame, width=5)
        nlr_r.pack(side=tk.LEFT, padx=(0, 5))
        nlr_r.insert(0, "2000")
        ttk.Label(nlr_frame, text="Nkin:").pack(side=tk.LEFT, padx=(0, 5))
        nlr_nkin = ttk.Entry(nlr_frame, width=5)
        nlr_nkin.pack(side=tk.LEFT, padx=(0, 5))
        nlr_nkin.insert(0, "2")

        # FeII component
        feii_frame = ttk.Frame(instance_frame)
        feii_frame.pack(fill=tk.X, padx=5, pady=2)
        use_feii = tk.BooleanVar(value=True)
        ttk.Checkbutton(feii_frame, text="FeII", variable=use_feii, 
                        command=lambda: self.toggle_component(feii_frame, use_feii.get())).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(feii_frame, text="velscale:").pack(side=tk.LEFT, padx=(0, 5))
        feii_velscale = ttk.Entry(feii_frame, width=5)
        feii_velscale.pack(side=tk.LEFT, padx=(0, 5))
        feii_velscale.insert(0, "10")
        ttk.Label(feii_frame, text="GH cont:").pack(side=tk.LEFT, padx=(0, 5))
        feii_gh_cont = ttk.Entry(feii_frame, width=5)
        feii_gh_cont.pack(side=tk.LEFT, padx=(0, 5))
        feii_gh_cont.insert(0, "2")
        ttk.Label(feii_frame, text="GH emis:").pack(side=tk.LEFT, padx=(0, 5))
        feii_gh_emis = ttk.Entry(feii_frame, width=5)
        feii_gh_emis.pack(side=tk.LEFT, padx=(0, 5))
        feii_gh_emis.insert(0, "0")

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda cf=instance_frame: self.delete_AGN_instance(cf))
        delete_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Update the instance dictionary
        self.agn_instances.append({
            'frame': instance_frame,
            'use_main_agn': use_main_agn,
            'igroup': agn_igroup,
            'id': agn_id,
            'name': agn_name,
            'iscalable': agn_scalable,
            'imodel': agn_imodel,
            'icloudy': agn_icloudy,
            'suffix': agn_suffix,
            'w_min': agn_w_min,
            'w_max': agn_w_max,
            'nw': agn_nw,
            'use_bbb': use_bbb,
            'bbb_frame': bbb_frame,
            'bbb_name': bbb_name,
            'bbb_w_min': bbb_w_min,
            'bbb_w_max': bbb_w_max,
            'bbb_nw': bbb_nw,
            'use_blr': use_blr,
            'blr_frame': blr_frame,
            'blr_file': blr_file,
            'blr_r': blr_r,
            'blr_nkin': blr_nkin,
            'use_nlr': use_nlr,
            'nlr_frame': nlr_frame,
            'nlr_file': nlr_file,
            'nlr_r': nlr_r,
            'nlr_nkin': nlr_nkin,
            'use_feii': use_feii,
            'feii_frame': feii_frame,
            'feii_velscale': feii_velscale,
            'feii_gh_cont': feii_gh_cont,
            'feii_gh_emis': feii_gh_emis
        })

        # Initialize the main AGN component visibility
        self.toggle_component(agn_params_frame, use_main_agn.get())

    # Add this new method to toggle component visibility
    def toggle_component(self, frame, state):
        for child in frame.winfo_children():
            if state:
                child.grid()
            else:
                child.grid_remove()

    # Update the generate_command method to include only selected components
    def generate_command(self):
        # ... (keep existing code)

        # AGN settings
        for agn in self.agn_instances:
            if agn['use_main_agn'].get():
                agn_igroup = agn['igroup'].get()
                agn_id = agn['id'].get()
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

            if agn['use_bbb'].get():
                bbb_id = int(agn_id) + 1
                command.extend([
                    "-bbb",
                    f"{bbb_id},{bbb_id},{agn['bbb_name'].get()},1,{agn['bbb_w_min'].get()},{agn['bbb_w_max'].get()},{agn['bbb_nw'].get()}"
                ])

            if agn['use_blr'].get():
                blr_id = int(agn_id) + 2
                command.extend([
                    "-ls1",
                    f"{blr_id},{blr_id},BLR,1,{agn['blr_file'].get()},{agn['blr_r'].get()},2,{agn['blr_nkin'].get()}"
                ])

            if agn['use_feii'].get():
                feii_id = int(agn_id) + 3
                command.extend([
                    "-k",
                    f"{feii_id},{feii_id},FeII,1,1,1,0,0,1,1,1",
                    "--kin",
                    f"{feii_id},{agn['feii_velscale'].get()},{agn['feii_gh_cont'].get()},{agn['feii_gh_emis'].get()}"
                ])

            if agn['use_nlr'].get():
                nlr_id = int(agn_id) + 4
                command.extend([
                    "-ls1",
                    f"{nlr_id},{nlr_id},NLR,1,{agn['nlr_file'].get()},{agn['nlr_r'].get()},2,{agn['nlr_nkin'].get()}"
                ])

        # ... (keep the rest of the method unchanged)

    # Update the apply_agn_settings method to handle the new structure
    def apply_agn_settings(self, settings):
        # ... (keep existing code)

        for instance_settings in settings:
            self.add_AGN_instance()
            instance = self.agn_instances[-1]
            for key, value in instance_settings.items():
                if key in instance and key != 'frame':
                    if key.startswith('use_'):
                        instance[key].set(value)
                        self.toggle_component(instance[f'{key[4:]}_frame'], value)
                    elif isinstance(instance[key], ttk.Entry):
                        instance[key].delete(0, tk.END)
                        instance[key].insert(0, value)
                    elif isinstance(instance[key], ttk.Combobox):
                        instance[key].set(value)

    # Update the get_agn_settings method to include the use_* flags
    def get_agn_settings(self):
        return [
            {key: (widget.get() if isinstance(widget, (ttk.Entry, ttk.Combobox)) else widget.get() if isinstance(widget, tk.BooleanVar) else None)
             for key, widget in instance.items() if key not in ['frame', 'bbb_frame', 'blr_frame', 'nlr_frame', 'feii_frame']}
            for instance in self.agn_instances
        ]

    def delete_AGN_instance(self, frame):
        for instance in self.agn_instances:
            if instance['frame'] == frame:
                self.agn_instances.remove(instance)
                break
        frame.destroy()

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)

    def generate_command(self):
        # Use sys.executable to get the path of the current Python interpreter
        command = [sys.executable, "-u", "bayesed.py"]  # Added '-u' for unbuffered output
        
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
            
            if all(ssp_values):
                command.extend(["-ssp", ",".join(ssp_values)])
                
            if all(sfh_values):
                command.extend(["--sfh", ",".join(sfh_values)])
                
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

        return command

    def run_bayesed(self):
        command = self.generate_command()
        
        self.update_output("Executing command: " + " ".join(command) + "\n")
        
        self.output_queue = queue.Queue()
        threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()
        self.master.after(100, self.check_output_queue)

    def execute_command(self, command):
        try:
            # Use subprocess.STARTUPINFO to hide console window on Windows
            startupinfo = None
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                             universal_newlines=True, bufsize=1, startupinfo=startupinfo)
            
            for line in iter(self.process.stdout.readline, ''):
                if self.stop_output_thread.is_set():
                    break
                self.output_queue.put(line)
            
            self.process.wait()
            
            # if self.process.returncode == 0:
            #     self.output_queue.put("BayeSED execution completed\n")
            # else:
            #     self.output_queue.put(f"BayeSED execution failed, return code: {self.process.returncode}\n")
        
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}\n")
        
        finally:
            self.output_queue.put(None)  # Signal that the process has finished

    def check_output_queue(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line is None:  # Process has finished
                    return
                self.update_output(line)
        except queue.Empty:
            self.master.after(100, self.check_output_queue)

    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.NORMAL)  # Keep it in normal state
        self.output_text.update_idletasks()  # Force update of the widget

    def browse_input_file(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.input_file.delete(0, tk.END)
            self.input_file.insert(0, filename)

    def browse_outdir(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.outdir.delete(0, tk.END)
            self.outdir.insert(0, dirname)

    def create_cosmology_tab(self):
        cosmology_frame = ttk.Frame(self.notebook)
        self.notebook.add(cosmology_frame, text="Cosmology")

        # Cosmology parameters
        cosmo_frame = ttk.Frame(cosmology_frame)
        cosmo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(cosmo_frame, variable=self.use_cosmology, 
                        command=lambda: self.toggle_widgets(list(self.cosmology_params.values()), self.use_cosmology.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)

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
                        command=lambda: self.toggle_widgets(self.igm_radiobuttons, self.use_igm.get()),
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)

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
                        command=self.toggle_redshift_widgets,
                        style='Large.TCheckbutton').pack(side=tk.LEFT, padx=5)

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
        filename = filedialog.askopenfilename()
        if filename:
            self.filters.delete(0, tk.END)
            self.filters.insert(0, filename)

    def browse_filters_selected(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.filters_selected.delete(0, tk.END)
            self.filters_selected.insert(0, filename)

    def export_settings(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json")
        if filename:
            settings = self.get_all_settings()
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)

    def import_settings(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, 'r') as f:
                settings = json.load(f)
            self.apply_all_settings(settings)

    def get_all_settings(self):
        settings = {
            "basic": self.get_basic_settings(),
            "galaxy": self.get_galaxy_settings(),
            "agn": self.get_agn_settings(),
            "cosmology": self.get_cosmology_settings(),
            "advanced": self.get_advanced_settings()
        }
        return settings

    def get_basic_settings(self):
        return {
            "input_file": self.input_file.get(),
            "input_type": self.input_type.get(),
            "outdir": self.outdir.get(),
            "verbose": self.verbose.get(),
            "filters": self.filters.get(),
            "filters_selected": self.filters_selected.get(),
            "priors_only": self.priors_only.get(),
            "no_photometry_fit": self.no_photometry_fit.get(),
            "no_spectra_fit": self.no_spectra_fit.get(),
            "use_snr": self.use_snr.get(),
            "snrmin1": self.snrmin1.get(),
            "snrmin2": self.snrmin2.get(),
            "use_sys_err": self.use_sys_err.get(),
            "sys_err_mod": [widget.get() for widget in self.sys_err_widgets[:5]],
            "sys_err_obs": [widget.get() for widget in self.sys_err_widgets[5:]],
            "save_bestfit": self.save_bestfit.get(),
            "save_bestfit_type": self.save_bestfit_type.get(),
            "save_sample_par": self.save_sample_par.get(),
            "save_sample_obs": self.save_sample_obs.get(),
            "save_pos_sfh": self.save_pos_sfh.get(),
            "save_pos_sfh_ngrid": self.save_pos_sfh_ngrid.get(),
            "save_pos_sfh_ilog": self.save_pos_sfh_ilog.get(),
            "save_pos_spec": self.save_pos_spec.get(),
            "save_sample_spec": self.save_sample_spec.get(),
            "save_summary": self.save_summary.get(),
            "output_mock_photometry": self.output_mock_photometry.get(),
            "output_mock_photometry_type": self.output_mock_photometry_type.get(),
            "output_mock_spectra": self.output_mock_spectra.get(),
            "output_model_absolute_magnitude": self.output_model_absolute_magnitude.get(),
            "output_pos_obs": self.output_pos_obs.get(),
            "use_build_sedlib": self.use_build_sedlib.get(),
            "build_sedlib": self.build_sedlib.get(),
            "unweighted_samples": self.unweighted_samples.get(),
            "use_sfr": self.use_sfr.get(),
            "sfr_myr": self.sfr_myr_entry.get(),
            "use_output_sfh": self.use_output_sfh.get(),
            "output_sfh_ntimes": self.output_sfh_ntimes.get(),
            "output_sfh_ilog": self.output_sfh_ilog.get(),
            "suffix": self.suffix.get()
        }

    def get_galaxy_settings(self):
        return [
            {
                "ssp": [widget.get() for widget in instance['ssp']],
                "sfh": [widget.get() for widget in instance['sfh']],
                "dal": [widget.get() for widget in instance['dal']],
                "dem": [widget.get() for widget in instance['dem']],
                "dem_model": instance['dem'][1].get(),  # Add this line to save the DEM model selection
                "dem_additional": [widget.get() for widget in self.additional_dem_widgets[instance['dem'][1].get()][1]],
                "ssp_id": instance['ssp_id'].get(),
                "sfh_id": instance['sfh_id'].get(),
                "dal_id": instance['dal_id'].get(),
                "dem_id": instance['dem_id'].get()
            }
            for instance in self.galaxy_instances
        ]

    def get_agn_settings(self):
        return [
            {key: widget.get() for key, widget in instance.items() if key != 'frame'}
            for instance in self.agn_instances
        ]

    def get_cosmology_settings(self):
        return {
            "use_cosmology": self.use_cosmology.get(),
            "cosmology_params": {param: widget.get() for param, widget in self.cosmology_params.items()},
            "use_igm": self.use_igm.get(),
            "igm_model": self.igm_model.get(),
            "use_redshift": self.use_redshift.get(),
            "redshift_params": {param: widget.get() for param, widget in self.redshift_params.items()}
        }

    def get_advanced_settings(self):
        return {
            "use_multinest": self.use_multinest.get(),
            "multinest_params": {param: widget.get() for param, widget in self.multinest_widgets.items()},
            "use_nnlm": self.use_nnlm.get(),
            "nnlm_params": {param: widget.get() for param, widget in self.nnlm_widgets.items()},
            "use_ndumper": self.use_ndumper.get(),
            "ndumper_params": {param: widget.get() for param, widget in self.ndumper_widgets.items()},
            "use_gsl": self.use_gsl.get(),
            "gsl_params": {param: widget.get() for param, widget in self.gsl_widgets.items()},
            "use_misc": self.use_misc.get(),
            "misc_params": {param: widget.get() for param, widget in self.misc_widgets.items()}
        }

    def apply_all_settings(self, settings):
        try:
            self.apply_basic_settings(settings.get("basic", {}))
            self.apply_galaxy_settings(settings.get("galaxy", []))
            self.apply_agn_settings(settings.get("agn", []))
            self.apply_cosmology_settings(settings.get("cosmology", {}))
            self.apply_advanced_settings(settings.get("advanced", {}))
            
            # Update the state of widgets after applying all settings
            self.toggle_widgets([self.snrmin1, self.snrmin2], self.use_snr.get())
            self.toggle_widgets(self.sys_err_widgets, self.use_sys_err.get())
            self.toggle_widgets([self.build_sedlib], self.use_build_sedlib.get())
            self.toggle_widgets([self.sfr_myr_entry], self.use_sfr.get())
            self.toggle_widgets([self.output_sfh_ntimes, self.output_sfh_ilog], self.use_output_sfh.get())
            self.save_bestfit_type.config(state="readonly" if self.save_bestfit.get() else "disabled")
            self.output_mock_photometry_type.config(state="readonly" if self.output_mock_photometry.get() else "disabled")
            
            # Force update of all frames
            self.master.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    def apply_basic_settings(self, settings):
        for key, value in settings.items():
            if hasattr(self, key):
                widget = getattr(self, key)
                if isinstance(widget, tk.Entry) or isinstance(widget, ttk.Entry) or isinstance(widget, ttk.Combobox):
                    widget.delete(0, tk.END)
                    widget.insert(0, value)
                elif isinstance(widget, tk.BooleanVar):
                    widget.set(value)
        
        # Handle special cases
        if "sys_err_mod" in settings:
            for widget, value in zip(self.sys_err_widgets[:5], settings["sys_err_mod"]):
                widget.delete(0, tk.END)
                widget.insert(0, value)
        if "sys_err_obs" in settings:
            for widget, value in zip(self.sys_err_widgets[5:], settings["sys_err_obs"]):
                widget.delete(0, tk.END)
                widget.insert(0, value)
        
        # Ensure all checkbox states are properly restored
        for key in [
            "priors_only", "no_photometry_fit", "no_spectra_fit", "use_snr", "use_sys_err",
            "save_bestfit", "save_sample_par", "save_sample_obs", "save_pos_sfh",
            "save_pos_spec", "save_sample_spec", "save_summary", "output_mock_photometry",
            "output_mock_spectra", "output_model_absolute_magnitude", "output_pos_obs",
            "use_build_sedlib", "unweighted_samples", "use_sfr", "use_output_sfh"
        ]:
            if key in settings:
                getattr(self, key).set(settings[key])

    def apply_galaxy_settings(self, settings):
        # Clear existing instances
        for instance in self.galaxy_instances:
            instance['frame'].destroy()
        self.galaxy_instances.clear()

        # Create new instances from settings
        for instance_settings in settings:
            try:
                self.add_galaxy_instance()
                instance = self.galaxy_instances[-1]
                for key in ['ssp', 'sfh', 'dal', 'dem']:
                    for widget, value in zip(instance[key], instance_settings[key]):
                        widget.delete(0, tk.END)
                        widget.insert(0, value)
                
                # Handle DEM model selection
                dem_model = instance_settings.get('dem_model', '0')  # Default to '0' if not found
                instance['dem'][1].set(dem_model)
                
                # Handle DEM additional parameters
                if dem_model in self.additional_dem_widgets:
                    additional_widgets = self.additional_dem_widgets[dem_model][1]
                    for widget, value in zip(additional_widgets, instance_settings['dem_additional']):
                        widget.delete(0, tk.END)
                        widget.insert(0, value)
                else:
                    print(f"Warning: DEM model {dem_model} not found in current configuration.")

                # Handle the ID settings
                for id_key in ['ssp_id', 'sfh_id', 'dal_id', 'dem_id']:
                    instance[id_key].delete(0, tk.END)
                    instance[id_key].insert(0, instance_settings.get(id_key, ''))

                # Update DEM parameters after setting the model
                self.update_dem_params(None, instance['frame'])
            except Exception as e:
                print(f"Error applying galaxy instance settings: {str(e)}")

    def apply_agn_settings(self, settings):
        # Clear existing instances
        for instance in self.agn_instances:
            instance['frame'].destroy()
        self.agn_instances.clear()

        # Create new instances from settings
        for instance_settings in settings:
            self.add_AGN_instance()
            instance = self.agn_instances[-1]
            for key, value in instance_settings.items():
                if key in instance and key != 'frame':
                    if key.startswith('use_'):
                        instance[key].set(value)
                        self.toggle_component(instance[f'{key[4:]}_frame'], value)
                    elif isinstance(instance[key], ttk.Entry):
                        instance[key].delete(0, tk.END)
                        instance[key].insert(0, value)
                    elif isinstance(instance[key], ttk.Combobox):
                        instance[key].set(value)

    def apply_cosmology_settings(self, settings):
        self.use_cosmology.set(settings.get("use_cosmology", False))
        for param, value in settings.get("cosmology_params", {}).items():
            if param in self.cosmology_params:
                widget = self.cosmology_params[param]
                widget.delete(0, tk.END)
                widget.insert(0, value)
        
        self.use_igm.set(settings.get("use_igm", False))
        self.igm_model.set(settings.get("igm_model", "1"))
        
        self.use_redshift.set(settings.get("use_redshift", False))
        for param, value in settings.get("redshift_params", {}).items():
            if param in self.redshift_params:
                widget = self.redshift_params[param]
                widget.delete(0, tk.END)
                widget.insert(0, value)

        # Update widget states
        self.toggle_widgets(list(self.cosmology_params.values()), self.use_cosmology.get())
        self.toggle_widgets(self.igm_radiobuttons, self.use_igm.get())
        self.toggle_redshift_widgets()

    def apply_advanced_settings(self, settings):
        for section in ['multinest', 'nnlm', 'ndumper', 'gsl', 'misc']:
            use_var = getattr(self, f'use_{section}')
            use_var.set(settings.get(f'use_{section}', False))
            
            params = settings.get(f'{section}_params', {})
            widgets = getattr(self, f'{section}_widgets')
            for param, value in params.items():
                if param in widgets:
                    widget = widgets[param]
                    widget.delete(0, tk.END)
                    widget.insert(0, value)
            
            # Update widget states
            self.toggle_widgets(widgets.values(), use_var.get())

    def update_dem_params(self, event, frame):
        imodel = event.widget.get() if event else frame.dem[1].get()
        
        # Find the correct instance
        instance = next((inst for inst in self.galaxy_instances if inst['frame'] == frame), None)
        if not instance:
            return  # If we can't find the instance, just return

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

