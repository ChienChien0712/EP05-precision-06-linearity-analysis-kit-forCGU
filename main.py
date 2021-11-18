import tkinter as tk
import tkinter.filedialog
from utils.EP06 import *
from utils.EP05 import *

window = tk.Tk()
window.title('EP05&EP06')
window.geometry('770x400')

end = 2000.0

def select_file():
    global filename
    filename = tkinter.filedialog.askopenfilename(filetypes = (("csv files","*.csv"),("xlsx files","*.xlsx")))
    if filename !="":
        if filename[-4:] == ".csv":
            data = pd.read_csv(filename)
        elif filename[-4:] == 'xlsx':
            data = pd.read_excel(filename)
        t.insert('end',data)
        t.insert('end','\n...')
        t.insert('end','\n>>>\n')
        t.see(end)

    else:
        t.insert('end',"No file were selected.")
        t.insert('end','\n>>>\n')
        t.see(end)

def NonlinearityAnalysis():
    try:
        data = regression_1to3(filename)
        x = e1.get().strip()
        y = e2.get().strip()
        nonlinearity_rate = float(e_Nonlinearity_Rate.get())
        nonlinearity_table = data.nonlinearity_error(x,y,nonlinearity_rate=nonlinearity_rate,plot=False)
        t.insert('end',nonlinearity_table)
        t.insert('end','\n>>>\n')
        t.see(end)
    except:
        t.insert('end',"KeyError. Please check the variable name!", 'warning')
        t.insert('end','\n>>>\n')
        t.see(end)



def NonlinearityAnalysisPlot():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()
    nonlinearity_rate = float(e_Nonlinearity_Rate.get())    
    nonlinearity_table = data.nonlinearity_error(x,y,nonlinearity_rate=nonlinearity_rate,plot=True)

def NonlinearityAnalysisSave():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()
    nonlinearity_rate = float(e_Nonlinearity_Rate.get())
    nonlinearity_table = data.nonlinearity_error(x,y,nonlinearity_rate=nonlinearity_rate,plot=False)
    savefilename =  tkinter.filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    if savefilename != '':
        if savefilename[-4:] !='.csv':
            savefilename +='.csv'
        nonlinearity_table.to_csv( savefilename,index=False)
        t.insert('end','file was saved as '+savefilename)
        t.insert('end','\n>>>\n')  
        t.see(end)  

def RepeatablityAnalysis():
    try:
        data = regression_1to3(filename)
        x = e1.get().strip()
        y = e2.get().strip()
        repeatability_table = data.repeatability_error(x,y,brief=True)
        t.insert('end',repeatability_table)
        t.insert('end','\n>>>\n')
        t.see(end)
    except:
        t.insert('end',"KeyError. Please check the variable name!", 'warning')
        t.insert('end','\n>>>\n')
        t.see(end)
def RepeatablityAnalysisSave():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()
    repeatability_table = data.repeatability_error(x,y,brief=False)
    savefilename =  tkinter.filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    if savefilename != '':
        if savefilename[-4:] !='.csv':
            savefilename +='.csv'
        repeatability_table.to_csv( savefilename,index=False)
        t.insert('end','file was saved as '+savefilename)
        t.insert('end','\n>>>\n')  
        t.see(end)  

def PlotRegression1():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()   
    data.plot_regression1to3(x,y,dim=1)
def PlotRegression2():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()   
    data.plot_regression1to3(x,y,dim=2)
def PlotRegression3():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()   
    data.plot_regression1to3(x,y,dim=3)  
def PlotRegression4():
    data = regression_1to3(filename)
    x = e1.get().strip()
    y = e2.get().strip()   
    data.plot_regression1to3(x,y,dim=4)   

def regression1to3_params_save():
    data = regression_1to3(filename)
    X = e1.get().strip()
    Y = e2.get().strip() 
    savefilename =  tkinter.filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    if savefilename != '':
        if savefilename[-4:] !='.csv':
            savefilename +='.csv'
        data.regression1to3_params_to_csv(X=X,Y=Y,filename=savefilename)
        t.insert('end','file was saved as '+savefilename)
        t.insert('end','\n>>>\n')  
        t.see(end)  

#####################################
def AnalysisforPrecision():
    try:
        data = VCA(filename)
        con = e_con.get().strip()
        day = e_day.get().strip()
        run = e_run.get().strip()
        result = data.nested_ANOVA('%s~%s/%s'%(con,day,run))
        t.insert('end',result)
        t.insert('end','\n>>>\n')   
        t.see(end)
    except KeyError:
        t.insert('end','KeyError!')
        t.insert('end','\n>>>\n')   
        t.see(end) 
    except NameError:
        t.insert('end','NameError!Please input a file to analysis.')
        t.insert('end','\n>>>\n')   
        t.see(end)               

def AnalysisforPrecisionSave():
    try:
        data = VCA(filename)
        con = e_con.get().strip()
        day = e_day.get().strip()
        run = e_run.get().strip()
        result = data.nested_ANOVA('%s~%s/%s'%(con,day,run),brief=False)
        savefilename =  tkinter.filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
        if savefilename != '':
            if savefilename[-4:] !='.csv':
                savefilename +='.csv'
            nonlinearity_table.to_csv( savefilename,index=False)
            t.insert('end','file was saved as '+savefilename)
            t.insert('end','\n>>>\n') 
            t.see(end)       
    except KeyError:
        t.insert('end','KeyError!')
        t.insert('end','\n>>>\n') 
        t.see(end) 
def AnalysisforPrecisionPlot():
    try:
        data = VCA(filename)
        con = e_con.get().strip()
        day = e_day.get().strip()
        data.plot_meanRun(x=day,y=con)   
    except KeyError:
        t.insert('end','KeyError!')
        t.insert('end','\n>>>\n')  
        t.see(end)

def AnalysisforPrecisionAll(): 
    data = VCA(filename)
    con = e_conAll.get().strip().split(',')
    day = e_day.get().strip()
    run = e_run.get().strip()

    savefilename =  tkinter.filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    if savefilename != '':
        if savefilename[-5:] !='.xlsx':
            savefilename +='.xlsx'
        writer = pd.ExcelWriter(savefilename, engine='xlsxwriter')
        for i in con:
            result = data.nested_ANOVA('%s~%s/%s'%(i,day,run),brief=False)
            result.to_excel(writer, sheet_name=i)
        writer.save()
        t.insert('end','file was saved as '+savefilename)
        t.insert('end','\n>>>\n') 
        t.see(end)
def ScatterAll():
    data = VCA(filename)
    con = e_conAll.get().strip().split(',')
    day = e_day.get().strip()
    run = e_run.get().strip()    
    data.scatter(day,con[0],allplot=True,allcon=con)      
def LineGraphAll():
    data = VCA(filename)
    con = e_conAll.get().strip().split(',')
    day = e_day.get().strip()
    run = e_run.get().strip()    
    data.plot_meanRun(day,con[0],allplot=True,allcon=con)      
#text
txtFrame = tk.Frame(window, borderwidth=1, relief="sunken")
t = tk.Text(txtFrame, wrap = tk.NONE,height=10,width= 105, borderwidth=0)
vscroll = tk.Scrollbar(txtFrame, orient=tk.VERTICAL, command=t.yview)
xscroll = tk.Scrollbar(txtFrame,orient= tk.HORIZONTAL,command=t.xview)
t['yscroll'] = vscroll.set
t['xscroll'] = xscroll.set
vscroll.pack(side="right", fill="y")
xscroll.pack(side="bottom",fill="x")
t.pack(side="left", fill="both", expand=True)
txtFrame.place(x=1, y=10)

#background
L_line = tk.Label(window,text="---------------------------------------------------------------------------------------------------------------------------------------",font='Arial')
L_line.place(x=0,y=240)
#選擇資料輸入按鈕
b_NA = tk.Button(window,text="Analysis of\n Nonlinearity",width=15,
               height=2,command=NonlinearityAnalysis)
b_NA.place(x=1,y=330)
b_NAplot = tk.Button(window,text="plot",width=8,
               height=1,command=NonlinearityAnalysisPlot)
b_NAplot.place(x=120,y=345)
b_NAsave = tk.Button(window,text="save to csv",width=13,
               height=1,command=NonlinearityAnalysisSave)
b_NAsave.place(x=185,y=345)

b_Repeatability = tk.Button(window,text="Analysis of\n Repeatablity",width=15,
               height=2,command=RepeatablityAnalysis)
b_Repeatability.place(x=300,y=280)
b_RepeatabilitySave = tk.Button(window,text="save to csv",width=13,
                    height=1,command=RepeatablityAnalysisSave)
b_RepeatabilitySave.place(x=300,y=330)
b_1st_plot = tk.Button(window,text="1st degree",width=10,
               height=1,command=PlotRegression1)
b_1st_plot.place(x=450,y=280)
b_2nd_plot = tk.Button(window,text="2nd degree",width=10,
               height=1,command=PlotRegression2)
b_2nd_plot.place(x=450,y=310)
b_3rd_plot = tk.Button(window,text="3rd degree",width=10,
               height=1,command=PlotRegression3)
b_3rd_plot.place(x=450,y=340)
b_4th_plot = tk.Button(window,text="4th degree",width=10,
               height=1,command=PlotRegression4)
b_4th_plot.place(x=450,y=370)
b_regression1to3_params_save = tk.Button(window,text="regression1to4 save",width=15,
               height=1,command=regression1to3_params_save)
b_regression1to3_params_save.place(x=550,y=280)

b_Precision = tk.Button(window,text="Analysis of\n Precision",width=15,
               height=2,command=AnalysisforPrecision)
b_Precision.place(x=220,y=190)
b_Psave = tk.Button(window,text="save to csv",width=13,
               height=1,command=AnalysisforPrecisionSave)
b_Psave.place(x=340,y=190)
b_Pplot = tk.Button(window,text="plot",width=13,
               height=1,command=AnalysisforPrecisionPlot)
b_Pplot.place(x=340,y=215)

b_PrecisionAll = tk.Button(window,text="One click to\n export results\n in all levels",width=15,
               height=3,command=AnalysisforPrecisionAll)
b_PrecisionAll.place(x=460,y=190)
b_scatterAll =  tk.Button(window,text="scatter plot",width=13,
               height=1,command=ScatterAll)
b_scatterAll.place(x=590,y=190)

b_LineGraphAll =  tk.Button(window,text="line graph",width=13,
               height=1,command=LineGraphAll)
b_LineGraphAll.place(x=590,y=220)
#Label
L1 = tk.Label(window,text="dilution factor")
L2 = tk.Label(window,text="column name of concentration")
L1.place(x=0,y=280)
L2.place(x=0,y=300)
L_Nonlinearity_Rate = tk.Label(window,text="Nonlinearity Rate",font=['Arial',9])
L_Nonlinearity_Rate.place(x=120,y=325)
L_vertical2 = tk.Label(window,text='| single level')
L_vertical2.place(x=60,y=165)
L_EP05 = tk.Label(window,text='EP05',font=['Arial',12])
L_EP05.place(x=0, y = 170)
L_EP06 = tk.Label(window,text='EP06',font=['Arial',12])
L_EP06.place(x=0, y = 260)
L_con = tk.Label(window,text='level name')
L_con.place(x=0,y=190)
L_day = tk.Label(window,text='column name of day')
L_day.place(x=0,y=210)
L_run = tk.Label(window,text='column name of run')
L_run.place(x=0,y=230)
L_multilevel = tk.Label(window,text='| multi-levels')
L_multilevel.place(x=450,y=165)

#Entry
e1 = tk.Entry(window,font='Arial',width=10)
e1.place(x=180,y=280)

e2 = tk.Entry(window,font='Arial',width=10)
e2.place(x=180,y=300)

e_con = tk.Entry(window,font='Arial',width=10)
e_con.place(x=120,y=190)

day_default = tk.StringVar(window, value='Day')
e_day = tk.Entry(window,textvariable=day_default,font='Arial',width=10)
e_day.place(x=120,y=210)

run_default = tk.StringVar(window, value='Run')
e_run = tk.Entry(window,textvariable=run_default,font='Arial',width=10)
e_run.place(x=120,y=230)

conAll_default = tk.StringVar(window, value='A,B,C,D')
e_conAll = tk.Entry(window,textvariable=conAll_default,width=10)
e_conAll.place(x=530,y=165)


Nonlinearity_Rate_default1 = tk.StringVar(window, value='0.05')
e_Nonlinearity_Rate = tk.Entry(window,textvariable=Nonlinearity_Rate_default1,width=8)
e_Nonlinearity_Rate.place(x=220,y=325)
#####menu#############
menubar = tk.Menu(window)
filemenu = tk.Menu(window,tearoff=0)
menubar.add_cascade(label="File",menu=filemenu)
filemenu.add_command(label='Open',command = select_file)
'''
editmenu = tk.Menu(window,tearoff=0)
menubar.add_cascade(label="Edit",menu=editmenu)
editmenu.add_command(label='Edit',command = select_file)
'''
window.config(menu=menubar)
#####################

window.mainloop()
