import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading

class BayeSEDGUI:
    def __init__(self, master):
        self.master = master
        master.title("BayeSED GUI")
        master.geometry("1400x800")
        self.galaxy_count = -1  # 从-1开始，这样第一个实例会是0
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=1, fill="both")

        self.create_basic_tab()
        self.create_galaxy_tab()
        self.create_output_tab()
        self.create_advanced_tab()

        self.create_output_frame()
        self.create_control_frame()

    def create_basic_tab(self):
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="基本设置")

        ttk.Label(basic_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_file = ttk.Entry(basic_frame, width=50)
        self.input_file.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(basic_frame, text="浏览", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(basic_frame, text="输入类型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_type = ttk.Combobox(basic_frame, values=["0 (flux in uJy)", "1 (AB magnitude)"])
        self.input_type.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.input_type.set("0 (flux in uJy)")

        ttk.Label(basic_frame, text="输出目录:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.outdir = ttk.Entry(basic_frame, width=50)
        self.outdir.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.outdir.insert(0, "result")
        ttk.Button(basic_frame, text="浏览", command=self.browse_outdir).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(basic_frame, text="详细程度:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.verbose = ttk.Combobox(basic_frame, values=["0", "1", "2", "3"])
        self.verbose.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.verbose.set("2")

    def create_galaxy_tab(self):
        self.galaxy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.galaxy_frame, text="Galaxy")

        self.galaxy_instances = []
        self.add_galaxy_instance()
        ttk.Button(self.galaxy_frame, text="添加Galaxy实例", command=self.add_galaxy_instance).pack(pady=5)

    def add_galaxy_instance(self):
        self.galaxy_count += 1
        instance_frame = ttk.LabelFrame(self.galaxy_frame, text=f"CSP {self.galaxy_count}")
        instance_frame.pack(fill=tk.X, padx=5, pady=5)

        # SSP设置
        ssp_frame = ttk.Frame(instance_frame)
        ssp_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ssp_frame, text="SSP:").grid(row=0, column=0, sticky=tk.W)
        
        ssp_params = [
            ("igroup", str(self.galaxy_count), 5),
            ("id", str(self.galaxy_count), 5),
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
            ttk.Label(ssp_frame, text=param+":").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
            if param in ['iscalable']:
                widget = ttk.Combobox(ssp_frame, values=["0", "1"], width=width)
                widget.set(default)
            else:
                widget = ttk.Entry(ssp_frame, width=width)
                widget.insert(0, default)
            widget.grid(row=0, column=2*i+2, padx=2)
            ssp_widgets.append(widget)

        # SFH设置
        sfh_frame = ttk.Frame(instance_frame)
        sfh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sfh_frame, text="SFH:").grid(row=0, column=0, sticky=tk.W)
        
        sfh_params = [
            ("id", str(self.galaxy_count), 5),
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
            ("itype_ceh", "0", 5, ["0: Default", "1: Alternative"])
        ]
        
        sfh_widgets = []
        for i, param_info in enumerate(sfh_params):
            param, default, width = param_info[:3]
            ttk.Label(sfh_frame, text=param+":").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
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

        # DAL设置
        dal_frame = ttk.Frame(instance_frame)
        dal_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(dal_frame, text="DAL:").grid(row=0, column=0, sticky=tk.W)
        
        dal_params = [
            ("id", str(self.galaxy_count), 5),
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
            ttk.Label(dal_frame, text=param+":").grid(row=0, column=2*i+1, sticky=tk.W, padx=2)
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

        self.galaxy_instances.append({
            'frame': instance_frame,
            'ssp': ssp_widgets,
            'sfh': sfh_widgets,
            'dal': dal_widgets
        })

    def create_output_tab(self):
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="输出设置")

        ttk.Label(output_frame, text="保存最佳拟合:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_bestfit = ttk.Combobox(output_frame, values=["0 (不保存)", "1 (hdf5)", "2 (fits和hdf5)"])
        self.save_bestfit.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.save_bestfit.set("0 (不保存)")

        self.save_sample_par = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="保存参数后验样本", variable=self.save_sample_par).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self.save_sample_obs = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="保存观测量后验样本", variable=self.save_sample_obs).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

    def create_advanced_tab(self):
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="高级设置")

        ttk.Label(advanced_frame, text="MultiNest设置:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.multinest_settings = ttk.Entry(advanced_frame, width=50)
        self.multinest_settings.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.multinest_settings.insert(0, "1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01")

    def create_output_frame(self):
        output_frame = ttk.LabelFrame(self.master, text="输出")
        output_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def create_control_frame(self):
        control_frame = ttk.Frame(self.master)
        control_frame.pack(padx=10, pady=10, fill=tk.X)

        self.run_button = ttk.Button(control_frame, text="运行", command=self.run_bayesed)
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
        
        # 基本设置
        input_type = self.input_type.get().split()[0]
        command.extend(["-i", f"{input_type},{self.input_file.get()}"])
        command.extend(["--outdir", self.outdir.get()])
        command.extend(["-v", self.verbose.get()])
        
        # galaxy实例设置
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
        
        # 输出设置
        command.extend(["--save_bestfit", self.save_bestfit.get().split()[0]])
        if self.save_sample_par.get():
            command.append("--save_sample_par")
        if self.save_sample_obs.get():
            command.append("--save_sample_obs")
        
        # 高级设置
        command.extend(["--multinest", self.multinest_settings.get()])
        
        return command

    def run_bayesed(self):
        command = self.generate_command()
        
        self.update_output("执行命令: " + " ".join(command) + "\n")
        
        threading.Thread(target=self.execute_command, args=(command,)).start()

    def execute_command(self, command):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            for line in process.stdout:
                self.update_output(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.update_output("BayeSED 运行完成\n")
            else:
                self.update_output(f"BayeSED 运行失败，返回码: {process.returncode}\n")
        
        except Exception as e:
            self.update_output(f"错误: {str(e)}\n")

    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

# 添加以下工具提示类
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
