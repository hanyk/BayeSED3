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
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=1, fill="both")

        self.create_basic_tab()
        self.create_galaxy_tab()
        self.create_AGN_tab()
        self.create_output_tab()
        self.create_advanced_tab()

        self.create_control_frame()
        self.create_output_frame()

    def create_basic_tab(self):
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="Basic Settings")

        ttk.Label(basic_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_file = ttk.Entry(basic_frame, width=50)
        self.input_file.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(basic_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(basic_frame, text="Input Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_type = ttk.Combobox(basic_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"])
        self.input_type.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.input_type.set("0 (flux in uJy)")

        ttk.Label(basic_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.outdir = ttk.Entry(basic_frame, width=50)
        self.outdir.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.outdir.insert(0, "result")
        ttk.Button(basic_frame, text="Browse", command=self.browse_outdir).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(basic_frame, text="Verbosity:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.verbose = ttk.Combobox(basic_frame, values=["0", "1", "2", "3"])
        self.verbose.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.verbose.set("2")

    def create_galaxy_tab(self):
        self.galaxy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.galaxy_frame, text="Galaxy")

        self.galaxy_instances = []
        self.add_galaxy_instance()
        ttk.Button(self.galaxy_frame, text="Add Galaxy Instance", command=self.add_galaxy_instance).pack(pady=5)

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
        
        instance_frame = ttk.LabelFrame(self.galaxy_frame, text=f"CSP {new_id}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        def update_ids(event):
            new_id = ssp_id_widget.get()
            sfh_id_widget.delete(0, tk.END)
            sfh_id_widget.insert(0, new_id)
            dal_id_widget.delete(0, tk.END)
            dal_id_widget.insert(0, new_id)

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

        # Add delete button
        delete_button = ttk.Button(instance_frame, text="Delete", command=lambda cf=instance_frame: self.delete_galaxy_instance(cf))
        delete_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.galaxy_instances.append({
            'frame': instance_frame,
            'ssp': ssp_widgets,
            'sfh': sfh_widgets,
            'dal': dal_widgets
        })

    def delete_galaxy_instance(self, frame):
        for instance in self.galaxy_instances:
            if instance['frame'] == frame:
                self.galaxy_instances.remove(instance)
                break
        frame.destroy()
        
        # No need to renumber instances, as each instance now has a unique id

    def create_output_tab(self):
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="Output Settings")

        ttk.Label(output_frame, text="Save Best Fit:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_bestfit = ttk.Combobox(output_frame, values=["0 (fits)", "1 (hdf5)", "2 (fits and hdf5)"])
        self.save_bestfit.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.save_bestfit.set("0 (fits)")

        self.save_sample_par = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Save Parameter Posterior Sample", variable=self.save_sample_par).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self.save_sample_obs = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Save Observable Posterior Sample", variable=self.save_sample_obs).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

    def create_advanced_tab(self):
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Advanced Settings")

        ttk.Label(advanced_frame, text="MultiNest Settings:").grid(row=0, column=0, columnspan=10, sticky=tk.W, padx=5, pady=5)

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
            row = i // 5 + 1
            col = (i % 5) * 2
            ttk.Label(advanced_frame, text=f"{param}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            widget = ttk.Entry(advanced_frame, width=8)
            widget.insert(0, default_values[i] if i < len(default_values) else "")
            widget.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.multinest_widgets[param] = widget
            CreateToolTip(widget, tooltip)

    def create_output_frame(self):
        output_frame = ttk.LabelFrame(self.master, text="Output")
        output_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def create_control_frame(self):
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=10)

        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_bayesed)
        self.run_button.pack(side=tk.LEFT, padx=5)

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

    def generate_command(self):
        command = ["python", "bayesed.py"]
        
        # Basic settings
        input_type = self.input_type.get().split()[0]
        command.extend(["-i", f"{input_type},{self.input_file.get()}"])
        command.extend(["--outdir", self.outdir.get()])
        command.extend(["-v", self.verbose.get()])
        
        # Galaxy instance settings
        for instance in self.galaxy_instances:
            ssp_values = [widget.get() for widget in instance['ssp']]
            sfh_values = [widget.get() for widget in instance['sfh']]
            dal_values = [widget.get() for widget in instance['dal']]
                
            if all(ssp_values):
                command.extend(["-ssp", ",".join(ssp_values)])
                
            if all(sfh_values):
                command.extend(["--sfh", ",".join(sfh_values)])
                
            if all(dal_values):
                command.extend(["--dal", ",".join(dal_values)])
        
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
        
        # Output settings
        command.extend(["--save_bestfit", self.save_bestfit.get().split()[0]])
        if self.save_sample_par.get():
            command.append("--save_sample_par")
        if self.save_sample_obs.get():
            command.append("--save_sample_obs")
        
        # Advanced settings
        multinest_values = [widget.get() for widget in self.multinest_widgets.values()]
        command.extend(["--multinest", ",".join(multinest_values)])
        
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

    def create_AGN_tab(self):
        agn_frame = ttk.Frame(self.notebook)
        self.notebook.add(agn_frame, text="AGN")

        # Container for AGN instances
        self.agn_instances_frame = ttk.Frame(agn_frame)
        self.agn_instances_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add AGN instance button
        ttk.Button(agn_frame, text="Add AGN Instance", command=self.add_AGN_instance).pack(pady=5)

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
                         background='yellow', relief='solid', borderwidth=1,
                         font=("times", "8", "normal"))
        label.pack(ipadx=1)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    gui = BayeSEDGUI(root)
    root.mainloop()