# Running into errors with the do_clustering() funtion

import VectorModelBuilder
from clusterer import do_clustering
from VectorModelBuilder import VectorModelBuilder
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory

VMB_ARGS = (("Dataset", ""),("Count Method", "ngram"),("Weighting", "ppmi"),("Output Directory","../vector_data/"),("Output File Name","None"),("Value of N", "3"))
CLUSTERER_ARGS = (("Stem of Files",""),("Output File","../vector_data/"),("V Scalar","1"),("Constrain Partition","False"),("Constrain PCS","False"))

LABELS = ["  Run  ","Browse...","False","True"]
FILE_TYPE = (("Text Files", "*.txt"),("All Files", "*.*"))
WEIGHT_METHODS = ["ppmi", "probability", "conditional_probability", "pmi", "none"]

# Two classes to help initiate the algorithm
class VectorModelBuilder_tk(VectorModelBuilder):
    def __init__(self,dataset,count_method,weighting,outdir,outfile,n):
        super().__init__(dataset,count_method,weighting,outdir,outfile,n)

    def run(self):
        self.create_vector_model()
        self.save_vector_model()

class Clusterer_tk():
    def __init__(self,input_file_stem, output_file, v_scalar, 
                  constrain_partition,
                  constrain_pcs):
        self.input_file_stem = input_file_stem
        self.output_file = output_file
        self.v_scalar = v_scalar
        self.constrain_partition = constrain_partition
        self.constrain_pcs = constrain_pcs

    def run(self):
        self.v_scalar = int(self.v_scalar)
        if (self.constrain_partition == "True"):
            self.constrain_partition = True
        else:
            self.constrain_partition = False

        if (self.constrain_pcs == "True"):
            self.constrain_pcs = True
        else:
            self.constrain_pcs = False

        do_clustering(self.input_file_stem,self.output_file,self.v_scalar,self.constrain_partition,self.constrain_pcs)

# Setting up the root of the window
window = tk.Tk()
window.title("Distributional Learning")
#window.columnconfigure(0,minsize=400,weight=1)
#window.rowconfigure(0,weight=1)

################### Vector Model Builder ################
vmb_frame = tk.Frame(master=window)
vmb_frame.grid(row=0,column=0,padx=10,pady=10,sticky="nswe")
vector_model_builder = tk.Label(master=vmb_frame,text="Vector Model Builder")
vector_model_builder.grid(row=0,column=0,sticky="w")
# looping to populate labels
j=1
for i, text in enumerate(VMB_ARGS):
    label = tk.Label(master=vmb_frame,text=VMB_ARGS[i][0])
    label.grid(row=j,column=0,sticky="e")
    j+=1

# Argument 1 Dataset
def open_arg1_browse():
    """Open a file."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"),("All Files", "*.*")]
    )
    if not filepath:
        return
    arg_path_ent.delete("0", tk.END)
    arg_path_ent.insert(tk.END, filepath)
arg_path_ent = tk.Entry(master=vmb_frame,width=40)
arg_path_ent.grid(row=1,column=1,sticky="w")
arg_path_ent.insert(0,VMB_ARGS[0][1])
arg1_browse = tk.Button(master=vmb_frame,text="Browse...",command=open_arg1_browse)
arg1_browse.grid(row=1,column=2,sticky="e")

# Argument 2 Count Method
method_ent = tk.Entry(master=vmb_frame)
method_ent.grid(row=2,column=1,sticky="w")
method_ent.insert(0,VMB_ARGS[1][1])

# Argument 3 Weighting arg
vmb_weight_var = tk.StringVar(vmb_frame)
    # add additional options for weighting here
vmb_weight_var.set(VMB_ARGS[2][1])
vmb_weight_menu = tk.OptionMenu(vmb_frame, vmb_weight_var,*WEIGHT_METHODS)
vmb_weight_menu.grid(row=3,column=1,sticky="w")

# Argument 4 Output directory
def open_outdir_browse():
    """Open a directory."""
    dirpath = askdirectory()
    if not dirpath:
        return
    outdir_ent.delete("0", tk.END)
    outdir_ent.insert(tk.END, dirpath)
outdir_ent = tk.Entry(master=vmb_frame,width=40)
outdir_ent.grid(row=4,column=1,sticky="w")
outdir_ent.insert(0,VMB_ARGS[3][1])
outdir_browse = tk.Button(master=vmb_frame,text="Browse...",command=open_outdir_browse)
outdir_browse.grid(row=4,column=2,sticky="e")

# Argument 5 Output File Name
outf_name_ent = tk.Entry(master=vmb_frame)
outf_name_ent.grid(row=5,column=1,sticky="w")
outf_name_ent.insert(0,VMB_ARGS[4][1])

# Argument 6 value of N "-n"
n_ent = tk.Entry(master=vmb_frame)
n_ent.grid(row=6,column=1,sticky="w")
n_ent.insert(0,VMB_ARGS[5][1])

# A dataset is required to run
def run_vector_model_builder():
    # Check for the default case of None for output file name
    outfile_arg = outf_name_ent.get()
    n_val = n_ent.get()
    if (outfile_arg == "None"):
        outfile_arg = None
    # Convert value of n from string to int or set the default to 3 if blank
    if (n_val == ""):
        n_val = 3
    else:
        n_val = int(n_val)
    Vector_tk = VectorModelBuilder_tk(arg_path_ent.get(),method_ent.get(),vmb_weight_var.get(),outdir_ent.get(),outfile_arg,n_val)
    Vector_tk.run()
run_VectorModelBuilder = tk.Button(master=vmb_frame,command=run_vector_model_builder,text=LABELS[0])
run_VectorModelBuilder.grid(row=7,column=0, sticky="w")
#########################################################################
############################### Clusterer ###############################
clusterer_frame = tk.Frame(master=window)
clusterer_frame.grid(row=1,column=0,padx=10,pady=10,sticky="nswe")

clusterer = tk.Label(master=clusterer_frame,text="Clusterer")
clusterer.grid(row=0,column=0,sticky="w")
# Looping to populate the labels
j=1
for i, text in enumerate(CLUSTERER_ARGS):
    label = tk.Label(master=clusterer_frame,text=CLUSTERER_ARGS[i][0])
    label.grid(row=j,column=0,sticky="e")
    j+=1

# Argument 1
file_name_ent = tk.Entry(master=clusterer_frame,width=40)
file_name_ent.grid(row=1,column=1,sticky="w")
file_name_ent.insert(0,CLUSTERER_ARGS[0][1])

# Argument 2 Output File
def output_file_browse():
    """Open a file."""
    dirpath = askdirectory()
    if not dirpath:
        return
    output_file_ent.delete("0", tk.END)
    output_file_ent.insert(tk.END, dirpath)
output_file_ent = tk.Entry(master=clusterer_frame,width=40)
output_file_ent.grid(row=2,column=1,sticky="w")
output_file_ent.insert(0,CLUSTERER_ARGS[1][1])
arg1_browse = tk.Button(master=clusterer_frame,text="Browse...",command=output_file_browse)
arg1_browse.grid(row=2,column=2,sticky="e")

# Argument 3 v_scalar
v_scalar_ent = tk.Entry(master=clusterer_frame)
v_scalar_ent.grid(row=3,column=1,sticky="w")
v_scalar_ent.insert(0,CLUSTERER_ARGS[2][1])

# Argument 4
constrain_partition_var = tk.StringVar(clusterer_frame)
constrain_partition_var.set(CLUSTERER_ARGS[3][1])
constrain_partition_menu  = tk.OptionMenu(clusterer_frame, constrain_partition_var,*LABELS[2:])
constrain_partition_menu.grid(row=4,column=1,sticky="w")

# Argument 5
constrain_pcs_var = tk.StringVar(clusterer_frame)
    # add additional options for weighting here
constrain_pcs_var.set(CLUSTERER_ARGS[4][1])
constrain_pcs_menu  = tk.OptionMenu(clusterer_frame, constrain_pcs_var,*LABELS[2:])
constrain_pcs_menu.grid(row=5,column=1,sticky="w")

# To run, the first two fields must be filled
def run_clusterer():
    do_clustering_tk = Clusterer_tk(file_name_ent.get(),output_file_ent.get(),v_scalar_ent.get(),constrain_partition_var.get(),constrain_pcs_var.get())
    do_clustering_tk.run()
run_clusterer_btn = tk.Button(master=clusterer_frame,command=run_clusterer,text=LABELS[0])
run_clusterer_btn.grid(row=6,column=0, sticky="w")
########################################################################
# To run both, required fields include Dataset and Output File Name
def run_both():
    run_vector_model_builder()
    name = outf_name_ent.get()
    if (name == "None") or (name == ""):
        ''' Print an Error '''
        print("Name of file is None\nCan not do clustering")
        return
    cluster = Clusterer_tk(name,outf_name_ent.get(),v_scalar_ent.get(),constrain_partition_var.get(),constrain_pcs_var.get())
    cluster.run()
run_all_btn = tk.Button(master=window,text="  Run Both  ",command=run_both)
run_all_btn.grid(row=2,column=0,pady=10)
window.mainloop()