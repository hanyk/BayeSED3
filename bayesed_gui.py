import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import subprocess
import threading
from PIL import Image, ImageDraw, ImageTk, ImageFont

class BayeSEDGUI:
    def __init__(self, master):
        self.master = master
        master.title("BayeSED GUI")
        master.geometry("1400x800")
        self.galaxy_count = -1  # Start from -1, so the first instance will be 0
        
        # Initialize instances lists
        self.galaxy_instances = []
        self.agn_instances = []
        self.cosmology_params = {}
        self.igm_model = tk.StringVar()
        self.redshift_params = {}
        
        # Create and set the icon
        self.create_icon()

        self.create_widgets()

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

        # 创建左侧框架用于输入设置
        left_frame = ttk.Frame(basic_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Input File
        ttk.Label(left_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_file = ttk.Entry(left_frame, width=50)
        self.input_file.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(left_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=2)

        # Input Type
        ttk.Label(left_frame, text="Input Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_type = ttk.Combobox(left_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"])
        self.input_type.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        self.input_type.set("0 (flux in uJy)")

        # Output Directory
        ttk.Label(left_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.outdir = ttk.Entry(left_frame, width=50)
        self.outdir.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        self.outdir.insert(0, "result")
        ttk.Button(left_frame, text="Browse", command=self.browse_outdir).grid(row=2, column=2, padx=5, pady=2)

        # Verbosity
        ttk.Label(left_frame, text="Verbosity:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.verbose = ttk.Combobox(left_frame, values=["0", "1", "2", "3"])
        self.verbose.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        self.verbose.set("2")

        # Filters
        ttk.Label(left_frame, text="Filters:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters = ttk.Entry(left_frame, width=50)
        self.filters.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(left_frame, text="Browse", command=self.browse_filters).grid(row=4, column=2, padx=5, pady=2)

        # Filters Selected
        ttk.Label(left_frame, text="Filters Selected:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.filters_selected = ttk.Entry(left_frame, width=50)
        self.filters_selected.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(left_frame, text="Browse", command=self.browse_filters_selected).grid(row=5, column=2, padx=5, pady=2)

        # 创建右侧框架用于保存和输出选项
        right_frame = ttk.Frame(basic_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Save Options
        save_frame = ttk.LabelFrame(right_frame, text="Save Options")
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
        output_frame = ttk.LabelFrame(right_frame, text="Output Options")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        # Output mock photometry
        self.output_mock_photometry = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Output Mock Photometry", variable=self.output_mock_photometry).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(output_frame.winfo_children()[-1], "Output mock photometry with best fit")

        self.output_mock_photometry_type = ttk.Combobox(output_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"], width=15, state="disabled")
        self.output_mock_photometry_type.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.output_mock_photometry_type.set("0 (flux in uJy)")
        CreateToolTip(self.output_mock_photometry_type, "Type of mock photometry output")

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

        # Suffix
        ttk.Label(output_frame, text="Suffix:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.suffix = ttk.Entry(output_frame, width=20)
        self.suffix.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        CreateToolTip(self.suffix, "Add suffix to the name of output file")

        # 创建底部框架用于运行按钮、导入/导出设置和输出
        bottom_frame = ttk.Frame(basic_frame)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # 控制按钮框架
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Run button
        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_bayesed)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Export and Import buttons
        ttk.Button(control_frame, text="Export Settings", command=self.export_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Import Settings", command=self.import_settings).pack(side=tk.LEFT, padx=5)

        # 输出框架
        output_frame = ttk.LabelFrame(bottom_frame, text="Output")
        output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Clear button
        clear_button = ttk.Button(output_frame, text="Clear", command=self.clear_output)
        clear_button.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure grid weights
        basic_frame.grid_columnconfigure(0, weight=1)
        basic_frame.grid_columnconfigure(1, weight=1)
        basic_frame.grid_rowconfigure(0, weight=1)
        basic_frame.grid_rowconfigure(1, weight=1)

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
        # Find the next available igroup and id
        used_ids = set()
        for instance in self.galaxy_instances:
            used_ids.add(instance['ssp'][0].get())  # igroup
            used_ids.add(instance['ssp'][1].get())  # id
        for instance in self.agn_instances:
            used_ids.add(instance['igroup'].get())
            used_ids.add(instance['id'].get())
        
        new_id = 0
        while str(new_id) in used_ids:
            new_id += 1
        
        instance_frame = ttk.LabelFrame(self.galaxy_instances_frame, text=f"CSP {new_id}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        def update_ids(event):
            new_id = ssp_id_widget.get()
            sfh_id_widget.delete(0, tk.END)
            sfh_id_widget.insert(0, new_id)
            dal_id_widget.delete(0, tk.END)
            dal_id_widget.insert(0, new_id)
            dem_id_widget.delete(0, tk.END)
            dem_id_widget.insert(0, new_id)

        # SSP settings
        ssp_frame = ttk.Frame(instance_frame)
        ssp_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ssp_frame, text="SSP:").grid(row=0, column=0, sticky=tk.W)
        
        ssp_params = [
            ("igroup", str(new_id), 5),
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

        # DEM settings
        dem_frame = ttk.LabelFrame(instance_frame, text="DEM")
        dem_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(dem_frame, text="DEM:").grid(row=0, column=0, sticky=tk.W)

        dem_params = [
            ("id", str(new_id), 3),
            ("imodel", "0", 3, [
                "0: Greybody",
                "1: Blackbody",
                "2: FANN",
                "3: AKNN"
            ]),
            ("iscalable", "-2", 3),
        ]

        dem_widgets = []
        for i, param_info in enumerate(dem_params):
            param, default, width = param_info[:3]
            ttk.Label(dem_frame, text=f"{param}:").grid(row=0, column=i*2+1, sticky=tk.E)
            if len(param_info) > 3:  # If there are options
                widget = ttk.Combobox(dem_frame, values=[opt.split(":")[0] for opt in param_info[3]], width=width)
                widget.set(default)
                tooltip = "\n".join(param_info[3])
                CreateToolTip(widget, tooltip)
                if param == "imodel":
                    widget.bind("<<ComboboxSelected>>", lambda event, f=dem_frame: self.update_dem_params(event, f))
            else:
                widget = ttk.Entry(dem_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=i*2+2, sticky=tk.W)
            dem_widgets.append(widget)
            if param == 'id':
                dem_id_widget = widget

        # Create a frame for extra DEM parameters
        extra_dem_frame = ttk.Frame(dem_frame)
        extra_dem_frame.grid(row=0, column=7, columnspan=10, sticky=tk.W)

        # Create the instance dictionary
        new_instance = {
            'frame': instance_frame,
            'ssp': ssp_widgets,
            'sfh': sfh_widgets,
            'dal': dal_widgets,
            'dem': dem_widgets,
            'dem_extra': [],
            'dem_extra_frame': extra_dem_frame
        }

        # Append the new instance to the list
        self.galaxy_instances.append(new_instance)

        # Now trigger the update_dem_params to show the initial extra parameters
        self.update_dem_params(type('Event', (), {'widget': dem_widgets[1]})(), dem_frame)

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda cf=instance_frame: self.delete_galaxy_instance(cf))
        delete_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # No need to renumber instances, as each instance now has a unique id

    def create_advanced_tab(self):
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Advanced Settings")

        # MultiNest Settings
        multinest_frame = ttk.LabelFrame(advanced_frame, text="MultiNest Settings")
        multinest_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        multinest_params = [
            ("INS", "Importance Nested Sampling flag"),
            ("mmodal", "Multimodal flag"),
            ("ceff", "Constant efficiency mode flag"),
            ("nlive", "Number of live points"),
            ("efr", "Sampling efficiency"),
            ("tol", "Tolerance"),
            ("updInt", "Update interval"),
            ("Ztol", "Evidence tolerance"),
            ("seed", "Random seed"),
            ("fb", "Feedback level"),
            ("resume", "Resume from a previous run"),
            ("outfile", "Write output files"),
            ("logZero", "Log of Zero"),
            ("maxiter", "Maximum number of iterations"),
            ("acpt", "Acceptance rate")
        ]

        default_values = "1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01".split(',')
        self.multinest_widgets = {}

        for i, (param, tooltip) in enumerate(multinest_params):
            row = i // 3
            col = i % 3 * 2
            ttk.Label(multinest_frame, text=f"{param}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(multinest_frame, width=8)
            widget.insert(0, default_values[i] if i < len(default_values) else "")
            widget.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.multinest_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # NNLM Settings
        nnlm_frame = ttk.LabelFrame(advanced_frame, text="NNLM Settings")
        nnlm_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        nnlm_params = [
            ("method", "Method (0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl)", "0"),
            ("Niter1", "Number of iterations for first step", "10000"),
            ("tol1", "Tolerance for first step", "0"),
            ("Niter2", "Number of iterations for second step", "10"),
            ("tol2", "Tolerance for second step", "0.01"),
            ("p1", "Parameter p1", "0.05"),
            ("p2", "Parameter p2", "0.95")
        ]
        self.nnlm_widgets = {}
        for i, (param, tooltip, default) in enumerate(nnlm_params):
            ttk.Label(nnlm_frame, text=f"{param}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(nnlm_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.nnlm_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Ndumper Settings
        ndumper_frame = ttk.LabelFrame(advanced_frame, text="Ndumper Settings")
        ndumper_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        ndumper_params = [
            ("max_number", "Maximum number", "1"),
            ("iconverged_min", "Minimum convergence flag", "0"),
            ("Xmin_squared_Nd", "Xmin^2/Nd value", "-1")
        ]
        self.ndumper_widgets = {}
        for i, (param, tooltip, default) in enumerate(ndumper_params):
            ttk.Label(ndumper_frame, text=f"{param}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(ndumper_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.ndumper_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # GSL Integration and Multifit Settings
        gsl_frame = ttk.LabelFrame(advanced_frame, text="GSL Settings")
        gsl_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        gsl_params = [
            ("integration_epsabs", "Integration absolute error", "0"),
            ("integration_epsrel", "Integration relative error", "0.1"),
            ("integration_limit", "Integration limit", "1000"),
            ("multifit_type", "Multifit type (ols or huber)", "ols"),
            ("multifit_tune", "Multifit tuning parameter", "1.0")
        ]
        self.gsl_widgets = {}
        for i, (param, tooltip, default) in enumerate(gsl_params):
            ttk.Label(gsl_frame, text=f"{param}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(gsl_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.gsl_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Other Miscellaneous Settings
        misc_frame = ttk.LabelFrame(advanced_frame, text="Other Settings")
        misc_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        misc_params = [
            ("NfilterPoints", "Number of filter points", "30"),
            ("Nsample", "Number of samples", ""),
            ("Ntest", "Number of objects for test run", ""),
            ("niteration", "Number of iterations", "0"),
            ("logZero", "Log of Zero", "-1e90"),
            ("lw_max", "Max line coverage in km/s", "10000")
        ]
        self.misc_widgets = {}
        for i, (param, tooltip, default) in enumerate(misc_params):
            row = i // 3
            col = i % 3 * 2
            ttk.Label(misc_frame, text=f"{param}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(misc_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.misc_widgets[param] = widget
            CreateToolTip(widget, tooltip)

        # Checkboxes for boolean options
        checkbox_frame = ttk.Frame(advanced_frame)
        checkbox_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        self.no_photometry_fit = tk.BooleanVar()
        ttk.Checkbutton(checkbox_frame, text="No photometry fit", variable=self.no_photometry_fit).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.no_spectra_fit = tk.BooleanVar()
        ttk.Checkbutton(checkbox_frame, text="No spectra fit", variable=self.no_spectra_fit).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.unweighted_samples = tk.BooleanVar()
        ttk.Checkbutton(checkbox_frame, text="Use unweighted samples", variable=self.unweighted_samples).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        # Configure grid weights
        advanced_frame.grid_columnconfigure(0, weight=1)
        advanced_frame.grid_columnconfigure(1, weight=1)

        # SFR_over
        sfr_frame = ttk.LabelFrame(advanced_frame, text="SFR Over")
        sfr_frame.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(sfr_frame, text="Past Myr1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sfr_past_myr1 = ttk.Entry(sfr_frame, width=10)
        self.sfr_past_myr1.insert(0, "10")
        self.sfr_past_myr1.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(sfr_frame, text="Past Myr2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.sfr_past_myr2 = ttk.Entry(sfr_frame, width=10)
        self.sfr_past_myr2.insert(0, "100")
        self.sfr_past_myr2.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # SNRmin1 and SNRmin2
        snr_frame = ttk.LabelFrame(advanced_frame, text="SNR Settings")
        snr_frame.grid(row=4, column=1, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(snr_frame, text="SNRmin1 (phot,spec):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.snrmin1 = ttk.Entry(snr_frame, width=10)
        self.snrmin1.insert(0, "0,0")
        self.snrmin1.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(snr_frame, text="SNRmin2 (phot,spec):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.snrmin2 = ttk.Entry(snr_frame, width=10)
        self.snrmin2.insert(0, "0,0")
        self.snrmin2.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # build_sedlib
        self.build_sedlib = tk.StringVar(value="0")
        ttk.Label(advanced_frame, text="Build SED Library:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(advanced_frame, text="Rest", variable=self.build_sedlib, value="0").grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(advanced_frame, text="Observed", variable=self.build_sedlib, value="1").grid(row=5, column=2, sticky=tk.W, padx=5, pady=2)

        # Confidence Levels
        ttk.Label(advanced_frame, text="Confidence Levels:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.cl = ttk.Entry(advanced_frame, width=15)
        self.cl.insert(0, "0.68,0.95")
        self.cl.grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)

        # Priors Only
        self.priors_only = tk.BooleanVar()
        ttk.Checkbutton(advanced_frame, text="Priors Only", variable=self.priors_only).grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)

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
        # Find the next available id
        used_ids = set()
        for instance in self.agn_instances:
            used_ids.add(int(instance['igroup'].get()))
            used_ids.add(int(instance['id'].get()))
        for instance in self.galaxy_instances:
            used_ids.add(int(instance['ssp'][0].get()))  # igroup
            used_ids.add(int(instance['ssp'][1].get()))  # id
        
        new_id = 0
        while new_id in used_ids:
            new_id += 1
        
        instance_frame = ttk.LabelFrame(self.agn_instances_frame, text=f"AGN {new_id}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)
    
        # Arrange sub-parameters horizontally with 5 parameters per row
        # First row parameters: igroup, id, name, iscalable, imodel
        ttk.Label(instance_frame, text="igroup:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        agn_igroup = ttk.Entry(instance_frame, width=8)
        agn_igroup.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        agn_igroup.insert(0, str(new_id))
        CreateToolTip(agn_igroup, "Group ID")
    
        ttk.Label(instance_frame, text="id:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        agn_id = ttk.Entry(instance_frame, width=8)
        agn_id.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        agn_id.insert(0, str(new_id))
        CreateToolTip(agn_id, "Model ID")
    
        ttk.Label(instance_frame, text="name:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        agn_name = ttk.Entry(instance_frame, width=12)
        agn_name.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        agn_name.insert(0, "AGN")
        CreateToolTip(agn_name, "Name of the AGN Model (default: AGN)")
    
        ttk.Label(instance_frame, text="iscalable:").grid(row=0, column=6, sticky=tk.W, padx=5, pady=2)
        agn_scalable = ttk.Combobox(instance_frame, values=["0", "1"], width=5)
        agn_scalable.grid(row=0, column=7, sticky=tk.W, padx=5, pady=2)
        agn_scalable.set("1")
        CreateToolTip(agn_scalable, "Is Scalable")
    
        ttk.Label(instance_frame, text="imodel:").grid(row=0, column=8, sticky=tk.W, padx=5, pady=2)
        agn_imodel = ttk.Combobox(instance_frame, values=["0 (qsosed)", "1 (agnsed)", "2 (fagnsed)", "3 (relagn)", "4 (relqso)", "5 (agnslim)"], width=12)
        agn_imodel.grid(row=0, column=9, sticky=tk.W, padx=5, pady=2)
        agn_imodel.set("0 (qsosed)")
        CreateToolTip(agn_imodel, "Model Subtype:\n0: qsosed\n1: agnsed\n2: fagnsed\n3: relagn\n4: relqso\n5: agnslim")
    
        # Second row parameters: icloudy, suffix, w_min, w_max, Nw
        ttk.Label(instance_frame, text="icloudy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        agn_icloudy = ttk.Combobox(instance_frame, values=["0", "1"], width=5)
        agn_icloudy.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        agn_icloudy.set("0")
        CreateToolTip(agn_icloudy, "Cloudy Model Flag")
    
        ttk.Label(instance_frame, text="suffix:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        agn_suffix = ttk.Entry(instance_frame, width=12)
        agn_suffix.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        agn_suffix.insert(0, "")  # Set suffix to empty by default
        CreateToolTip(agn_suffix, "Suffix for the Model Name")
    
        ttk.Label(instance_frame, text="w_min:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        agn_w_min = ttk.Entry(instance_frame, width=8)
        agn_w_min.grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
        agn_w_min.insert(0, "300.0")
        CreateToolTip(agn_w_min, "Minimum Wavelength")
    
        ttk.Label(instance_frame, text="w_max:").grid(row=1, column=6, sticky=tk.W, padx=5, pady=2)
        agn_w_max = ttk.Entry(instance_frame, width=8)
        agn_w_max.grid(row=1, column=7, sticky=tk.W, padx=5, pady=2)
        agn_w_max.insert(0, "1000.0")
        CreateToolTip(agn_w_max, "Maximum Wavelength")
    
        ttk.Label(instance_frame, text="Nw:").grid(row=1, column=8, sticky=tk.W, padx=5, pady=2)
        agn_nw = ttk.Entry(instance_frame, width=5)
        agn_nw.grid(row=1, column=9, sticky=tk.W, padx=5, pady=2)
        agn_nw.insert(0, "200")
        CreateToolTip(agn_nw, "Number of Wavelength Points")
    
        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda cf=instance_frame: self.delete_AGN_instance(cf))
        delete_button.grid(row=1, column=10, rowspan=2, padx=5, pady=5, sticky=tk.N)
    
        # Add the instance to the list
        self.agn_instances.append({
            'frame': instance_frame,
            'igroup': agn_igroup,
            'id': agn_id,
            'name': agn_name,
            'iscalable': agn_scalable,
            'imodel': agn_imodel,
            'icloudy': agn_icloudy,
            'suffix': agn_suffix,
            'w_min': agn_w_min,
            'w_max': agn_w_max,
            'nw': agn_nw
        })

    def delete_AGN_instance(self, frame):
        for instance in self.agn_instances:
            if instance['frame'] == frame:
                self.agn_instances.remove(instance)
                break
        frame.destroy()
        
        # No need to renumber instances or reset self.next_agn_id

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def generate_command(self):
        command = ["python", "bayesed.py"]
        
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
            dem_extra_values = [widget.get() for widget in instance['dem_extra']]
            
            if all(ssp_values):
                command.extend(["-ssp", ",".join(ssp_values)])
                
            if all(sfh_values):
                command.extend(["--sfh", ",".join(sfh_values)])
                
            if all(dal_values):
                command.extend(["--dal", ",".join(dal_values)])
            
            if all(dem_values):
                imodel = dem_values[1]  # The imodel value
                igroup = ssp_values[0]  # Use SSP's igroup for DEM
                if imodel == "0":  # Greybody
                    command.extend(["--greybody", f"{igroup},{dem_values[0]},{dem_extra_values[0]},{dem_values[2]},{dem_extra_values[1]},{dem_extra_values[2]},{dem_extra_values[3]},{dem_extra_values[4]}"])
                elif imodel == "1":  # Blackbody
                    command.extend(["--blackbody", f"{igroup},{dem_values[0]},{dem_extra_values[0]},{dem_values[2]},{dem_extra_values[1]},{dem_extra_values[2]},{dem_extra_values[3]}"])
                elif imodel == "2":  # FANN
                    command.extend(["-a", f"{igroup},{dem_values[0]},{dem_extra_values[0]},{dem_values[2]}"])
                elif imodel == "3":  # AKNN
                    command.extend(["-k", f"{igroup},{dem_values[0]},{dem_extra_values[0]},{dem_values[2]},{dem_extra_values[1]},{dem_extra_values[2]},{dem_extra_values[3]},{dem_extra_values[4]},{dem_extra_values[5]},{dem_extra_values[6]},{dem_extra_values[7]}"])
        
        # AGN settings
        for agn in self.agn_instances:
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
        
        # Advanced settings
        multinest_values = [widget.get() for widget in self.multinest_widgets.values()]
        command.extend(["--multinest", ",".join(multinest_values)])
        
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
        cosmo_values = [f"{param}={widget.get()}" for param, widget in self.cosmology_params.items()]
        if cosmo_values:
            command.extend(["--cosmology", ",".join(cosmo_values)])

        # Add IGM model
        command.extend(["--IGM", self.igm_model.get()])

        # Add redshift parameters if enabled
        if self.use_redshift.get():
            redshift_values = [widget.get() for widget in self.redshift_params.values()]
            if all(redshift_values):
                command.extend(["-z", ",".join(redshift_values)])

        # NNLM settings
        nnlm_values = [widget.get() for widget in self.nnlm_widgets.values()]
        if all(nnlm_values):
            command.extend(["--NNLM", ",".join(nnlm_values)])

        # Ndumper settings
        ndumper_values = [widget.get() for widget in self.ndumper_widgets.values()]
        if all(ndumper_values):
            command.extend(["--Ndumper", ",".join(ndumper_values)])

        # GSL settings
        gsl_integration_values = [self.gsl_widgets[p].get() for p in ["integration_epsabs", "integration_epsrel", "integration_limit"]]
        if all(gsl_integration_values):
            command.extend(["--gsl_integration_qag", ",".join(gsl_integration_values)])
        
        gsl_multifit_values = [self.gsl_widgets[p].get() for p in ["multifit_type", "multifit_tune"]]
        if all(gsl_multifit_values):
            command.extend(["--gsl_multifit_robust", ",".join(gsl_multifit_values)])

        # Other miscellaneous settings
        for param, widget in self.misc_widgets.items():
            value = widget.get()
            if value:
                command.extend([f"--{param}", value])

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
        
        command.extend(["--SFR_over", f"{self.sfr_past_myr1.get()},{self.sfr_past_myr2.get()}"])
        command.extend(["--SNRmin1", self.snrmin1.get()])
        command.extend(["--SNRmin2", self.snrmin2.get()])
        
        if self.build_sedlib.get() != "0":
            command.extend(["--build_sedlib", self.build_sedlib.get()])
        
        command.extend(["--cl", self.cl.get()])
        
        if self.priors_only.get():
            command.append("--priors_only")

        return command

    def run_bayesed(self):
        command = self.generate_command()
        
        self.update_output("Executing command: " + " ".join(command) + "\n")
        
        threading.Thread(target=self.execute_command, args=(command,)).start()

    def execute_command(self, command):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            for line in process.stdout:
                self.update_output(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.update_output("BayeSED execution completed\n")
            else:
                self.update_output(f"BayeSED execution failed, return code: {process.returncode}\n")
        
        except Exception as e:
            self.update_output(f"Error: {str(e)}\n")

    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

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
        cosmo_frame = ttk.LabelFrame(cosmology_frame, text="Cosmology Parameters")
        cosmo_frame.pack(fill=tk.X, padx=5, pady=5)

        cosmo_params = [
            ("H0", "Hubble constant (km/s/Mpc)", "70"),
            ("omigaA", "Omega Lambda", "0.7"),
            ("omigam", "Omega Matter", "0.3")
        ]

        for i, (param, tooltip, default) in enumerate(cosmo_params):
            ttk.Label(cosmo_frame, text=f"{param}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(cosmo_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.cosmology_params[param] = widget
            CreateToolTip(widget, tooltip)

        # IGM model
        igm_frame = ttk.LabelFrame(cosmology_frame, text="IGM Model")
        igm_frame.pack(fill=tk.X, padx=5, pady=5)

        self.igm_model = tk.StringVar(value="1")
        igm_options = [
            ("0", "None"),
            ("1", "Madau (1995) model"),
            ("2", "Meiksin (2006) model"),
            ("3", "hyperz"),
            ("4", "FSPS"),
            ("5", "Inoue+2014")
        ]

        for i, (value, text) in enumerate(igm_options):
            ttk.Radiobutton(igm_frame, text=text, variable=self.igm_model, value=value).grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)

        # Redshift parameters
        redshift_frame = ttk.LabelFrame(cosmology_frame, text="Redshift Parameters (Optional)")
        redshift_frame.pack(fill=tk.X, padx=5, pady=5)

        self.use_redshift = tk.BooleanVar(value=False)
        ttk.Checkbutton(redshift_frame, text="Set Redshift Parameters", variable=self.use_redshift, command=self.toggle_redshift_widgets).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        redshift_params = [
            ("iprior_type", "Prior type (0-7)", "1"),
            ("is_age", "Age-dependent flag (0 or 1)", "0"),
            ("min", "Minimum redshift", "0"),
            ("max", "Maximum redshift", "10"),
            ("nbin", "Number of bins", "100")
        ]

        self.redshift_widgets = []
        for i, (param, tooltip, default) in enumerate(redshift_params):
            ttk.Label(redshift_frame, text=f"{param}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(redshift_frame, width=10)
            widget.insert(0, default)
            widget.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            widget.config(state="disabled")
            self.redshift_params[param] = widget
            CreateToolTip(widget, tooltip)
            self.redshift_widgets.append(widget)

    def toggle_redshift_widgets(self):
        state = "normal" if self.use_redshift.get() else "disabled"
        for widget in self.redshift_widgets:
            widget.config(state=state)

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
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            command = self.generate_command()
            with open(filename, 'w') as f:
                f.write(" ".join(command))

    def import_settings(self):
        filename = filedialog.askopenfilename()
        if filename:
            with open(filename, 'r') as f:
                settings = f.read().split()
            # Here you would need to parse the settings and update the GUI accordingly
            # This is a complex task and would require a separate method to handle it

    def update_dem_params(self, event, frame):
        imodel = event.widget.get()
        
        # Find the correct instance
        instance = next((inst for inst in self.galaxy_instances if inst['frame'] == frame.master), None)
        if not instance:
            return  # If we can't find the instance, just return

        extra_frame = instance['dem_extra_frame']

        # Clear all existing widgets in extra_frame
        for widget in extra_frame.winfo_children():
            widget.destroy()

        instance['dem_extra'] = []

        extra_params = []
        if imodel == "0":  # Greybody
            extra_params = [
                ("name", "greybody", 15),
                ("ithick", "0", 3),
                ("w_min", "1", 3),
                ("w_max", "1000", 5),
                ("Nw", "200", 3)
            ]
        elif imodel == "1":  # Blackbody
            extra_params = [
                ("name", "blackbody", 15),
                ("w_min", "1", 3),
                ("w_max", "1000", 5),
                ("Nw", "200", 3)
            ]
        elif imodel == "2":  # FANN
            extra_params = [
                ("name", "FANN", 15)
            ]
        elif imodel == "3":  # AKNN
            extra_params = [
                ("name", "AKNN", 15),
                ("k", "1", 3),
                ("f_run", "1", 3),
                ("eps", "0", 3),
                ("iRad", "0", 3),
                ("iprep", "0", 3),
                ("Nstep", "1", 3),
                ("alpha", "0", 3)
            ]

        new_extra_widgets = []

        for i, (param, default, width) in enumerate(extra_params):
            label = ttk.Label(extra_frame, text=f"{param}:")
            label.grid(row=0, column=i*2, sticky=tk.E, padx=(0, 2))
            widget = ttk.Entry(extra_frame, width=width)
            widget.insert(0, default)
            widget.grid(row=0, column=i*2+1, sticky=tk.W, padx=(0, 5))
            new_extra_widgets.append(widget)

        # Update the instance with new extra widgets
        instance['dem_extra'] = new_extra_widgets

        # Force the frame to update its layout
        extra_frame.update_idletasks()

    def delete_galaxy_instance(self, frame):
        for instance in self.galaxy_instances:
            if instance['frame'] == frame:
                # Remove the extra frame if it exists
                if 'dem_extra_frame' in instance:
                    instance['dem_extra_frame'].destroy()
                self.galaxy_instances.remove(instance)
                break
        frame.destroy()

# Add the following tooltip class if not already present
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)

    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background='#FFFFDD', relief='solid', borderwidth=1,
                         font=("Arial", "12", "bold"))  # Increased font size to 12, changed to Arial and bold
        label.pack(ipadx=5, ipady=5)  # Added more padding

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    gui = BayeSEDGUI(root)
    root.mainloop()