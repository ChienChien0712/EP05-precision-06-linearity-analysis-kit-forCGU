import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

class regression_1to3:
    def __init__(self,data_name):
        if data_name[-4:] == '.csv':
            self.data = pd.read_csv(data_name)
        if data_name[-4:] == 'xlsx':
            self.data = pd.read_excel(data_name)    
        
    def regression1to3(self,X='X',Y='pg/mL(Log5PL)'):
        data_x = self.data[X]
        data_y = self.data[Y]
        
        #一次
        X = sm.add_constant(data_x)
        Y = data_y 
        model = sm.OLS(Y,X)
        results_1 = model.fit()
        SE_regression_1 = np.sqrt(sum((results_1.resid-np.mean(results_1.resid))**2)/results_1.df_resid)
        
        #二次
        X2 = np.column_stack((data_x, data_x**2))
        X2 = sm.add_constant(X2)
        model = sm.OLS(Y,X2)
        results_2 = model.fit()
        SE_regression_2 = np.sqrt(sum((results_2.resid-np.mean(results_2.resid))**2)/results_2.df_resid)
        
        #三次
        X3 = np.column_stack((data_x, data_x**2, data_x**3))
        X3 = sm.add_constant(X3)
        model = sm.OLS(Y,X3)
        results_3 = model.fit()
        SE_regression_3 = np.sqrt(sum((results_3.resid-np.mean(results_3.resid))**2)/results_3.df_resid)

        #四次
        X4 = np.column_stack((data_x, data_x**2, data_x**3,data_x**4))
        X4 = sm.add_constant(X4)
        model = sm.OLS(Y,X4)
        results_4 = model.fit()
        SE_regression_4 = np.sqrt(sum((results_4.resid-np.mean(results_4.resid))**2)/results_4.df_resid)
        
        return results_1, results_2, results_3, results_4, SE_regression_1, SE_regression_2, SE_regression_3, SE_regression_4
    def plot_regression1to3(self,X='X',Y='pg/mL(Log5PL)',dim=3):
        x = np.arange(0.5,max(self.data[X].unique())+1.5)
        Y1 = self.regression1to3(X,Y)[0].params[0]+ x*self.regression1to3(X,Y)[0].params[1]
        Y2 = self.regression1to3(X,Y)[1].params[0]+ x*self.regression1to3(X,Y)[1].params[1] +self.regression1to3(X,Y)[1].params[2]*x**2
        Y3 = self.regression1to3(X,Y)[2].params[0]+ x*self.regression1to3(X,Y)[2].params[1] +self.regression1to3(X,Y)[2].params[2]*x**2 + self.regression1to3(X,Y)[2].params[3]*x**3
        Y4 = self.regression1to3(X,Y)[3].params[0]+ x*self.regression1to3(X,Y)[3].params[1] +self.regression1to3(X,Y)[3].params[2]*x**2 + self.regression1to3(X,Y)[3].params[3]*x**3 + self.regression1to3(X,Y)[3].params[4]*x**4

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(self.data[X],self.data[Y],color='black')
        if dim ==3:
            ax1.plot(x,Y1) 
            ax1.plot(x,Y2) 
            ax1.plot(x,Y3)
            ax1.legend(['1st','2nd','3rd'])
        elif dim ==2:
            ax1.plot(x,Y1) 
            ax1.plot(x,Y2) 
            ax1.legend(['1st','2nd'])            
        elif dim ==1:
            ax1.plot(x,Y1)  
            ax1.legend(['1st'])  
        elif dim ==4:
            ax1.plot(x,Y1) 
            ax1.plot(x,Y2) 
            ax1.plot(x,Y3)
            ax1.plot(x,Y4)
            ax1.legend(['1st','2nd','3rd','4th'])                           
        
        plt.xticks(sorted(list(set(self.data[X]))),sorted(list(set(self.data[X]))),rotation=0,fontsize='small')
        plt.grid(linestyle='--', which='major',color='gray')
        plt.xlabel('concentration')
        plt.ylabel('result')
        plt.show()
    
    def repeatability_error(self,X='X',Y='pg/mL(Log5PL)',brief=False):
        mean_list = []
        diff_list = []
        diff2_list = []
        diff2_percent_list = []
        replicate = int(len(self.data)/len(set(self.data[X])))
        level = int(len(self.data)/replicate)
        level_value = sorted(list(set(self.data[X])))
        
        #Mean
        for i in level_value:
            ri = np.mean(self.data[self.data[X]==i][Y])
            mean_list.append(ri)

        #diff[[]]
        for i in level_value:
            ri = np.mean(self.data[self.data[X]==i][Y])
            diff = []
            for j in range(replicate):
                diff.append(list(self.data[self.data[X]==i][Y])[j]-ri)
            diff_list.append(diff)
        diff_list = np.array(diff_list)
        diff_list = diff_list.tolist()

        #diff2[]
        for i in level_value:
            ri = np.mean(self.data[self.data[X]==i][Y])
            diff2 = 0
            for j in range(replicate):
                diff2 += (list(self.data[self.data[X]==i][Y])[j]-ri)**2
            diff2_list.append(diff2)


        #%diff[[]]
        ri_list = []
        for i in level_value:
            ri_list.append([np.mean(self.data[self.data[X]==i][Y])]*replicate)
        ri_list = np.array(ri_list)
        diff_percent_list = np.array(diff_list)*100 / ri_list 
        diff_percent_list = diff_percent_list.tolist()

        #%diff2 []
        for i in level_value:
            ri = np.mean(self.data[self.data[X]==i][Y])
            diff2_percent = 0
            for j in range(replicate):
                diff2_percent += ((list(self.data[self.data[X]==i][Y])[j]-ri)*100 / ri)**2
            diff2_percent_list.append(diff2_percent)
            
        ####建立dataframe
        diff_list_mean = []
        for i in np.asmatrix(diff_list).T:
            diff_list_mean.append(np.mean(i))

        diff_percent_list_mean = []
        for i in np.asmatrix(diff_percent_list).T:
            diff_percent_list_mean.append(np.mean(i))

        if brief == False:
            data_output_dic = {'Concentration':level_value + ['pool','SD or CV%'],
                               'Mean':mean_list + [np.mean(mean_list),''],
                               'Difference':diff_list + [diff_list_mean,''],
                               'Squared Difference':diff2_list + [np.mean(diff2_list),np.sqrt(sum(diff2_list)/(level*(replicate-1)))],
                               '%Diff':diff_percent_list + [diff_percent_list_mean,''],
                               '%Squared Diff':diff2_percent_list + [np.mean(diff2_percent_list),np.sqrt(sum(diff2_percent_list)/(level*(replicate-1)))]}

            columns=['Concentration','Mean']
            for i in range(int(replicate)):
                diff_mat = np.asmatrix(diff_list)
                diff_percent_mat = np.asmatrix(diff_percent_list)
                data_output_dic['Difference%s'%(i+1)]=list(np.asarray(diff_mat.T[i])[0]) + [np.mean(list(np.asarray(diff_mat.T[i])[0])),'']
                data_output_dic['%%Diff%s'%(i+1)]=list(np.asarray(diff_percent_mat.T[i])[0]) + [np.mean(list(np.asarray(diff_percent_mat.T[i])[0])),'']
            for i in range(int(replicate)):
                columns.append('Difference%s'%(i+1))
            columns += ['Squared Difference']
            for i in range(int(replicate)):
                columns.append('%%Diff%s'%(i+1))
            columns += ['%Squared Diff']
        elif brief == True:
            data_output_dic = {'Concentration':level_value + ['pool','SD or CV%'],
                               'Mean':list(map(float,['%.1f' %i for i in mean_list])) + [round(np.mean(mean_list),1),''],
                               'Squared Difference':list(map(float,['%.2f' %i for i in diff2_list])) + [round(np.mean(diff2_list),2),round(np.sqrt(sum(diff2_list)/(level*(replicate-1))),2)],
                               '%Squared Diff':list(map(float,['%.2f' %i for i in diff2_percent_list])) + [round(np.mean(diff2_percent_list),2),round(np.sqrt(sum(diff2_percent_list)/(level*(replicate-1))),2)]}

            columns=['Concentration','Mean']
            for i in range(int(replicate)):
                diff_mat = np.asmatrix(diff_list)
                diff_percent_mat = np.asmatrix(diff_percent_list)
                data_output_dic['Difference%s'%(i+1)]=list(map(float,['%.2f' %i for i in list(np.asarray(diff_mat.T[i])[0])])) + [round(np.mean(list(np.asarray(diff_mat.T[i])[0])),2),'']
                data_output_dic['%%Diff%s'%(i+1)]=list(map(float,['%.2f' %i for i in list(np.asarray(diff_percent_mat.T[i])[0])])) + [round(np.mean(list(np.asarray(diff_percent_mat.T[i])[0])),2),'']
            for i in range(int(replicate)):
                columns.append('Difference%s'%(i+1))
            columns += ['Squared Difference']
            for i in range(int(replicate)):
                columns.append('%%Diff%s'%(i+1))
            columns += ['%Squared Diff']            

        SD_table = pd.DataFrame(data_output_dic, columns=columns)
        return SD_table
    
    def nonlinearity_error(self,X='X',Y='pg/mL(Log5PL)',nonlinearity_rate = 0.05,plot=True):
        Actual_mean = []
        Y1_list = []
        Y2_list = []
        replicate = int(len(self.data)/len(set(self.data[X])))
        level_value = sorted(list(set(self.data[X])))
        
        for i in level_value:
            ri = np.mean(self.data[self.data[X]==i][Y])
            Actual_mean.append(ri)


        for x in level_value:
            Y1 = self.regression1to3(X,Y)[0].params[0]+ x*self.regression1to3(X,Y)[0].params[1]
            Y2 = self.regression1to3(X,Y)[1].params[0]+ x*self.regression1to3(X,Y)[1].params[1] +self.regression1to3(X,Y)[1].params[2]*x**2
            Y1_list.append(Y1)
            Y2_list.append(Y2)
        Y2_Y1 = np.array(Y2_list)-np.array(Y1_list)
        Diff_percent = Y2_Y1/np.array(Y1_list)*100

        Regression_Diff ={'Actual':['Mean']+Actual_mean,
                          'Predicted_1':['1st-order']+Y1_list,
                          'Predicted_2':['2nd-order']+Y2_list,
                          'Difference':['2nd-1st']+Y2_Y1.tolist(),
                          '%Difference':['']+Diff_percent.tolist()
        }

        columns=['Actual','Predicted_1','Predicted_2','Difference','%Difference']

        Regression_Diff_table = pd.DataFrame(Regression_Diff, columns=columns,)
        
        #非線性誤差：5%
        if plot == True:
            x = Actual_mean
            limit_y_highgoal = np.array(Actual_mean)*nonlinearity_rate
            limit_y_lowgoal = np.array(Actual_mean)*nonlinearity_rate*-1

            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            ax1.scatter(x,Y2_Y1,color='black',marker='D')
            ax1.plot(x,limit_y_highgoal,linestyle='--',color='black') 
            ax1.plot(x,limit_y_lowgoal,linestyle='-.',color='black') 

            #plt.xticks(sorted(list(set(data_x))),sorted(list(set(data_x))),rotation=0,fontsize='small')
            ax1.legend(['High goal','Low goal','Difference'])

            plt.grid(linestyle='--', which='major',color='gray')
            plt.xlabel('Value')
            plt.ylabel('Difference')
            plt.show()
        return Regression_Diff_table    
    def regression1to3_params_to_csv(self,X='X',Y='pg/mL(Log5PL)',filename=False):
        results = self.regression1to3(X,Y)
        results_1 = results[0]
        results_2 = results[1]
        results_3 = results[2]
        results_4 = results[3]
        SE_regression_1 = results[4] 
        SE_regression_2 = results[5] 
        SE_regression_3 = results[6] 
        SE_regression_4 = results[7] 

        f1 = open(filename,'w')
        f1.write('Order,Coef. Symbol,Coefficient Value, Coefficient SE,t-test,df,Std Error Regression\n')

        f1.write('First,beta 0,%s,%s,%s\n'%(results_1.params[0],results_1.bse[0],results_1.tvalues[0]))
        f1.write('First,beta 1,%s,%s,%s,'%(results_1.params[1],results_1.bse[1],results_1.tvalues[1]))
        f1.write('%d,%s\n'%(results_1.df_resid,SE_regression_1))
        f1.write("\n")
        f1.write('Second,beta 0,%s,%s,%s\n'%(results_2.params[0],results_2.bse[0],results_2.tvalues[0]))
        f1.write('Second,beta 1,%s,%s,%s\n'%(results_2.params[1],results_2.bse[1],results_2.tvalues[1]))
        f1.write('Second,beta 2,%s,%s,%s,'%(results_2.params[2],results_2.bse[2],results_2.tvalues[2]))
        f1.write('%d,%s\n'%(results_2.df_resid,SE_regression_2))
        f1.write("\n")
        f1.write('Third,beta 0,%s,%s,%s\n'%(results_3.params[0],results_3.bse[0],results_3.tvalues[0]))
        f1.write('Third,beta 1,%s,%s,%s\n'%(results_3.params[1],results_3.bse[1],results_3.tvalues[1]))
        f1.write('Third,beta 2,%s,%s,%s\n'%(results_3.params[2],results_3.bse[2],results_3.tvalues[2]))
        f1.write('Third,beta 3,%s,%s,%s,'%(results_3.params[3],results_3.bse[3],results_3.tvalues[3]))
        f1.write('%d,%s\n'%(results_3.df_resid,SE_regression_3))
        f1.write("\n")
        f1.write('Forth,beta 0,%s,%s,%s\n'%(results_4.params[0],results_4.bse[0],results_4.tvalues[0]))
        f1.write('Forth,beta 1,%s,%s,%s\n'%(results_4.params[1],results_4.bse[1],results_4.tvalues[1]))
        f1.write('Forth,beta 2,%s,%s,%s\n'%(results_4.params[2],results_4.bse[2],results_4.tvalues[2]))
        f1.write('Forth,beta 3,%s,%s,%s\n'%(results_4.params[3],results_4.bse[3],results_4.tvalues[3]))
        f1.write('Forth,beta 4,%s,%s,%s,'%(results_4.params[4],results_4.bse[4],results_4.tvalues[4]))
        f1.write('%d,%s\n'%(results_4.df_resid,SE_regression_4))
        f1.close()        
