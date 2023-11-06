import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import cryptpandas as crp


class EBITDA:

    def __init__(self, pwd, map_loc):

        # Read table data
        self.read_table(pwd, map_loc)

        # Standard input parameters
        self.inputs = {}
        self.inputs['yr_prod_vol'] = 0 # Tons per year
        self.inputs['nb_yr_prod']  = 0 # Nb. of years in production
        self.inputs['price_factor'] = 1 # Rescale prices of elements by a factor 
        self.inputs['conc_factor'] = 1 # Rescale concentrations of elements by a factor
        self.inputs['Penalty']     = 0 # [USD/ton] cost for unwanted metals, type in 0 to 100
        self.inputs['RefRecovery'] = 0 # typically 0.85
        self.inputs['ProsEx']      = 0 # [USD/ton] cost of processing, low 10 USD/ton
        self.inputs['Processing']  = 0 # up-concentration before refining, will reduce refinement cost.
        self.inputs['RefCost']     = 0 # [USD/Ton](variable, default 70 USD/ton)
        self.inputs['CapEx']       = 0 # [USD]  (type in, default 1013 mUSD)
        self.inputs['ExpEnvEx']    = 0 # [USD] (type in, default 105 mUSD)
        self.inputs['OpEx_fix']    = 0 # [USD/yr] (default 200 mUSD)
        self.inputs['OpEx_var']    = 0  # [USD/ton/yr] variable with ore production
        self.inputs['Include_CO2_Trapping'] = False,
        self.inputs['Ultramafic_fraction'] = 0.25 # Fraction of ultramafic in rock tailing
        self.inputs['Ultramafic_reactive'] = 0.2  # Fraction reactive to carbonation (CO2 trapping)
        self.inputs['CO2_price'] = 100   # Price of stored CO2 [USD/ton]
        self.inputs['Magnesite_price'] = 500   # Price of magnesite [USD/ton] 
        self.inputs['Magnesite_factor'] = 1  # rescale Magnesite production by a factor
        self.inputs['selected_location'] = 7 # select number for "Data locations" above
        # Select all elements as default 
        self.selected_elements()


    def read_table(self, pwd, map_loc):
        # Elements    
        list_e = ( ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si','P',
                    'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
                    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut',
                    'Fl', 'Uup', 'Lv', 'Uus', 'Uuo'] ) 
        list_n = ( list(range(1, 16)) + list(range(16, 31)) + list(range(31, 45))
                + list(range(45, 59)) + list(range(59, 73)) + list(range(73, 87))
                + list(range(87, 101)) + list(range(101, 114)) + list(range(114, 119)) )
        map_n2e = dict(zip(list_n,list_e))
        # map on locations
        self.map_loc = map_loc
        # read file
        self.tbl_all = crp.read_encrypted(path='file', password=pwd)
        # change column names
        self.tbl_all = self.tbl_all.rename( columns={"a": "element", "b": "concentration [wt]", "c": "value [USD/m2]", "d": "value [USD/Ton]",  "e": "location"} )
        # Update elements
        elems_ = [map_n2e[ie] for ie in self.tbl_all['element'].values]
        self.tbl_all['element'] = elems_
        # output the different locations 
        print('Data locations:   ')
        for i, k in self.map_loc.items():
            print( i, ':', k , '   ')



    def update(self,inputs_dict):
        self.inputs.update(inputs_dict)
        if self.inputs['selected_location']:
            if 'selected_elements' in inputs_dict.keys():
                len_check = sum(self.tbl['element'].isin(self.inputs['selected_elements']))
                if len_check<len(self.inputs['selected_elements']):
                    print('Selected elements not (all) in dataset! Please, select from the list below:')
                    self.selected_elements()
                    self.print_elements()
                    sys.exit()
            else:
                self.selected_elements()
                self.print_elements()


    def selected_elements(self):
        self.tbl = self.tbl_all[self.tbl_all['location']==self.inputs['selected_location']]
        self.inputs['selected_elements'] = list(self.tbl['element'].unique())


    def print_elements(self):
        all_elements = list(self.tbl['element'].unique())
        loc_ = self.inputs['selected_location']
        print('Dataset',loc_)
        print('All elements in ', self.map_loc[loc_],':\n',all_elements) #self.inputs['selected_elements'])


    def profit(self):
        self.Profit = ( self.Net_ore_value_sum 
                      - self.Penalty_sum
                      - self.RefLoss_sum 
                      - self.RefCost_sum
                      - self.ProsEx_sum
                      - self.inputs['CapEx']
                      - self.inputs['ExpEnvEx']
                      - self.OpEx_sum )
        
        if self.inputs['Include_CO2_Trapping']==True:
            self.Profit += self.CO2_value_sum + self.Magnesite_value_sum

    def return_on_investment(self):
        return_ = ( self.Net_ore_value_sum 
                      - self.Penalty_sum
                      - self.RefLoss_sum 
                      - self.RefCost_sum
                      - self.ProsEx_sum
                      - self.OpEx_sum )
        investment_ =  self.inputs['CapEx'] + self.inputs['ExpEnvEx']
        self.Return_on_Inv = (return_ - investment_)/investment_
    

    def variable_sums(self, nb_yr_prod, ore_val=''):

        # value of selected elements
        column_name = [icol for icol in self.tbl.columns if '[USD/Ton]' in icol][0]

        # sanity check if all the selected elements are present a the location
        #self.selected_elements()
        len_check = sum(self.tbl['element'].isin(self.inputs['selected_elements']))
        if len_check<len(self.inputs['selected_elements']):
            print('Selected elements not (all) in dataset!')
            self.selected_elements()
            self.print_elements()
            exit()
        else:
            self.selected_elements_USD_Ton = np.array([ self.tbl[self.tbl['element']==ie][column_name].values[0] for ie in self.inputs['selected_elements'] ])

        # Compute variables
        self.total_vol = self.inputs['yr_prod_vol'] * nb_yr_prod
        # Element value can be increase/decrease by "price_factor" and "conc_factor"
        self.Elements_value = ( self.selected_elements_USD_Ton * self.total_vol 
                               * self.inputs['price_factor'] * self.inputs['conc_factor'] )

        if ore_val:
            self.Net_ore_value_sum = ore_val * self.total_vol
        else:
            self.Net_ore_value_sum = np.sum(self.Elements_value)
        
        self.Penalty_sum = (self.inputs['Penalty'] * self.total_vol)
        self.RefLoss_sum = (self.Net_ore_value_sum-self.Penalty_sum) * (1-self.inputs['RefRecovery'])
        self.RefCost_sum = self.inputs['Processing'] * (self.inputs['RefCost']*self.total_vol)
        self.ProsEx_sum = self.inputs['ProsEx'] * self.total_vol
        self.OpEx_sum = ( self.inputs['OpEx_fix'] + self.inputs['OpEx_var']*self.inputs['yr_prod_vol'] ) * nb_yr_prod


    def CO2_capture(self):

        CO2_prod = ( ( self.inputs['yr_prod_vol']*self.inputs['nb_yr_prod']) 
                    *self.inputs['Ultramafic_fraction']
                    *self.inputs['Ultramafic_reactive'] )
        self.CO2_value_sum = CO2_prod * self.inputs['CO2_price'] * self.inputs['price_factor']
        weight_CO2_to_Magnesite = 2
        reaction_conversion_CO2_to_Magnesite = 1/2 
        self.Magnesite_value_sum = ( CO2_prod * reaction_conversion_CO2_to_Magnesite * weight_CO2_to_Magnesite 
                                    * self.inputs['Magnesite_factor']
                                    * self.inputs['Magnesite_price']
                                    * self.inputs['price_factor'] ) 

    def waterfall(self):

        # Call functions
        #self.selected_elements()
        self.variable_sums(nb_yr_prod = self.inputs['nb_yr_prod'])
        
        if self.inputs['Include_CO2_Trapping']==True:
            # Run CO2 capture and magnesite production
            self.CO2_capture()
            self.profit()

            # Make waterfall plotly plot
            measure = list(np.repeat('relative', len(self.inputs['selected_elements']) + 9)) + ['total']
            x = ( self.inputs['selected_elements'] 
                + ['Penalty', 'RefLoss', 'RefCost', 'ProsEx', 'CapEx', 'ExpEnvEx', 'OpEx'] 
                + ['CO2 seq.', 'Magnesite']
                + ['Profit'])
            y = ( list(self.Elements_value) 
                + list([-self.Penalty_sum, -self.RefLoss_sum, -self.RefCost_sum, -self.ProsEx_sum, 
                        -self.inputs['CapEx'], -self.inputs['ExpEnvEx'], -self.OpEx_sum] ) 
                + list([self.CO2_value_sum, self.Magnesite_value_sum] )
                + list([self.Profit]) )
        else:
            self.profit()
            # Make waterfall plotly plot
            measure = list(np.repeat('relative', len(self.inputs['selected_elements']) + 7)) + ['total']
            x = ( self.inputs['selected_elements'] 
                + ['Penalty', 'RefLoss', 'RefCost', 'ProsEx', 'CapEx', 'ExpEnvEx', 'OpEx', 'Profit'])
            y = ( list(self.Elements_value) 
                + list([-self.Penalty_sum, -self.RefLoss_sum, -self.RefCost_sum, -self.ProsEx_sum, 
                        -self.inputs['CapEx'], -self.inputs['ExpEnvEx'], -self.OpEx_sum, self.Profit]) )            

        y = np.array(y)/1e6 # Plot in mUSD

        color_profit = "#2E8B57" # green if positive
        if self.Profit<0:
            color_profit = "#FF2400" # red if negative

        fig = go.Figure(go.Waterfall(
            measure = measure,
            x = x,
            textposition = 'auto' , #"outside",
            text = np.round(np.array(y)),
            y = y,
            width=1+0*y,
            increasing = {"marker":{"color":"#4682B4"}},
            decreasing = {"marker":{"color":"#e67300"}},
            totals = {"marker":{"color":color_profit}},
        ))

        fig.update_layout(
                title = ( '<b>'+self.map_loc[self.inputs['selected_location']]+'</b>'
                        +'<br>   Production of '+str(self.inputs['yr_prod_vol']/1e6)+' mTon/yr during '+str(self.inputs['nb_yr_prod'])+ ' yrs' ),
                yaxis_title="<b>mUSD<b>",
                showlegend = False,
                plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            zerolinecolor='black',
            zerolinewidth=2
        )
        fig.show()

        # save fig object to save later
        self.fig = fig


    def profit_over_time(self, low_yr=1, high_yr=10):
        
        Nb_yr_prod  = np.linspace(low_yr, high_yr, 100)   # Nb. of years in production
        Profits = np.zeros_like(Nb_yr_prod)
        for i, nb_yr_prod in enumerate(Nb_yr_prod):
            self.variable_sums(nb_yr_prod = nb_yr_prod)
            if self.inputs['Include_CO2_Trapping']==True:
                self.CO2_capture()
                self.profit()
            else:
                self.profit()
            Profits[i] = self.Profit
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Nb_yr_prod, y=Profits, name='Profit'))

        fig.update_layout(
                title = ( '<b>'+self.map_loc[self.inputs['selected_location']]+'</b>'
                        +'<br>   Production of '+str(self.inputs['yr_prod_vol']/1e6)+' mTon/yr during '+str(high_yr)+ ' yrs' ),
                xaxis_title="<b>Number of years<b>",
                yaxis_title="<b>Profit [USD]<b>",
                showlegend = False,
                plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            zerolinecolor='black',
            zerolinewidth=2
        )
        fig.show()

        # save fig object to save later
        self.fig = fig


    def return_on_investment_over_time(self, low_yr=1, high_yr=10):
        
        Nb_yr_prod  = np.linspace(low_yr, high_yr, 100)   # Nb. of years in production
        RonI = np.zeros_like(Nb_yr_prod)
        for i, nb_yr_prod in enumerate(Nb_yr_prod):
            self.variable_sums(nb_yr_prod = nb_yr_prod)
            self.return_on_investment()

            RonI[i] = self.Return_on_Inv
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Nb_yr_prod, y=RonI, name='Profit'))

        fig.update_layout(
                title = ( '<b>'+self.map_loc[self.inputs['selected_location']]+'</b>'
                        +'<br>   Production of '+str(self.inputs['yr_prod_vol']/1e6)+' mTon/yr during '+str(high_yr)+ ' yrs' ),
                xaxis_title="<b>Number of years<b>",
                yaxis_title="<b>Return on investment<b>",
                showlegend = False,
                plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            zerolinecolor='black',
            zerolinewidth=2
        )
        fig.show()

        # save fig object to save later
        self.fig = fig


    def BarPlot_Value_Systems(self, Selection_elements=[]):
   
        import plotly.express as px
        df = pd.DataFrame()
        for i in range(1,8):
            loc_ = self.map_loc[i].split("-")[0] +'<br>'+ self.map_loc[i].split("-")[1]
            tbl_ = self.tbl_all[self.tbl_all['location']==i][['value [USD/Ton]', 'element']]
            tbl_ = tbl_[tbl_['element'].isin(Selection_elements[i])]
            tbl_['location_name'] = '<b>'+loc_+'<b>'
            df = pd.concat([df, tbl_])

        fig = go.Figure()
        for i, iel in enumerate(df['element'].unique()):  
            fig.add_trace(go.Bar(x=df[df['element']==iel]['location_name'],
                                y=df[df['element']==iel]['value [USD/Ton]'],
                                name=iel))

        fig.update_layout(barmode='stack',
                        yaxis_title="<b>Estimated value [USD/ton<b>]",)
        fig.show()  

        # save fig object to save later
        self.fig = fig



    def Profit_RonI_xNbYear_yOreVal(self, low_yr=1, high_yr=10, low_ore_val=100, high_ore_val=700, save_pngs=''):
        
        Nb_yr_prod  = np.linspace(low_yr, high_yr, 20)   # Nb. of years in production
        Ore_val  = np.linspace(low_ore_val, high_ore_val, 20)   # Ore value

        RonI = np.zeros( (len(Nb_yr_prod), len(Ore_val)) )
        Pro = np.zeros( (len(Nb_yr_prod), len(Ore_val)) )
        for i, nb_yr_prod in enumerate(Nb_yr_prod):
            for j, ore_val in enumerate(Ore_val):
                self.variable_sums(nb_yr_prod = nb_yr_prod, ore_val=ore_val)
                self.return_on_investment()
                RonI[i,j] = self.Return_on_Inv
                if self.inputs['Include_CO2_Trapping']==True:
                    self.CO2_capture()
                    self.profit()
                else:
                    self.profit()
                Pro[i,j] = self.Profit



        fig = go.Figure(data = go.Contour(x=Nb_yr_prod, y=Ore_val, z=Pro.transpose(),
                                          contours=dict(start=0,end=int(Pro.max()),size=round(Pro.max()/1e9/15)*1e9, showlabels = True),
                                          colorbar=dict(title='[USD]'),colorscale='greens'))
        #fig.add_contour(x=Nb_yr_prod, y=Ore_val, z=Pro.transpose(), contours_coloring='lines',
        #                contours=dict(start=int(Pro.min()/1e9)*1e9,end=0,size=1e9, showlabels = True),
        #                line_width=2,showscale=False)
        fig.add_contour(x=Nb_yr_prod, y=Ore_val, z=Pro.transpose(), contours_coloring='lines',
                        line_width=2,colorscale='Electric',contours=dict(start=0,
                                                                         end=int(Pro.max())+1,
                                                                         size=int(Pro.max())+1,
                                                                         showlabels = True), 
                                                                         showscale=False)
        fig.update_layout(title = ( '<b>Profit </b><br>'+'assuming a production of ' 
                                    + str(self.inputs['yr_prod_vol']/1e6) + ' mTonOre/yr' ),
                          xaxis_title="<b>Number of year [yr]<b>",
                          yaxis_title="<b>Ore Value [USD/ton]<b>",
                          showlegend = False, plot_bgcolor='white')
        fig.update_xaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey' )
        fig.update_yaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey',
                         zerolinecolor='black',zerolinewidth=2)
        fig.show()

        # save fig object to png
        if save_pngs:
            self.fig = fig
            self.savefig_png(save_pngs+'_Profit_'+str(self.inputs['yr_prod_vol']/1e6) + 'mTonOreYr', width=550, height=500)

        fig = go.Figure(data = go.Contour(x=Nb_yr_prod, y=Ore_val, z=RonI.transpose(),
                                          contours=dict(start=0,end=int(RonI.max()),size=int(RonI.max()/15),showlabels = True),
                                          colorbar=dict(title='[factor]'),colorscale='greens'))
        #fig.add_contour(x=Nb_yr_prod, y=Ore_val, z=RonI.transpose(), contours_coloring='lines', line=dict(color='red'),
        #                contours=dict(start=int(RonI.min()),end=0,size=1, showlabels = True),
        #                line_width=2,showscale=False)
        fig.add_contour(x=Nb_yr_prod, y=Ore_val, z=RonI.transpose(), contours_coloring='lines',
                        line_width=2,colorscale='Electric',contours=dict(start=0,
                                                                         end=int(RonI.max())+1,
                                                                         size=int(RonI.max())+1,
                                                                         showlabels = True), 
                                                                         showscale=False)
        fig.update_layout(title = ( '<b>Return on investment </b><br>'+'assuming a production of ' 
                                   + str(self.inputs['yr_prod_vol']/1e6)+ ' mTonOre/yr' ),
                          xaxis_title="<b>Number of year [yr]<b>",
                          yaxis_title="<b>Ore Value [USD/ton]<b>",
                          showlegend = False, plot_bgcolor='white')
        fig.update_xaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey' )
        fig.update_yaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey',
                         zerolinecolor='black',zerolinewidth=2)
        fig.show()

        # save fig object to png
        if save_pngs:
            self.fig = fig
            self.savefig_png(save_pngs+'_RonI_'+str(self.inputs['yr_prod_vol']/1e6) + 'mTonOreYr', width=550, height=500)


        return Pro, RonI

    def Profit_RonI_xTonYr_yOreVal(self, low_t_yr=1, high_t_yr=10, low_ore_val=100, high_ore_val=700, save_pngs=''):
        
        Ton_yr_prod  = np.linspace(low_t_yr, high_t_yr, 20)   # Nb. of years in production
        Ore_val  = np.linspace(low_ore_val, high_ore_val, 20)   # Ore value
        
        RonI = np.zeros( (len(Ton_yr_prod), len(Ore_val)) )
        Pro = np.zeros( (len(Ton_yr_prod), len(Ore_val)) )
        for i, ton_yr_prod in enumerate(Ton_yr_prod):
            for j, ore_val in enumerate(Ore_val):
                self.inputs['yr_prod_vol'] = ton_yr_prod
                self.variable_sums(nb_yr_prod = self.inputs['nb_yr_prod'], ore_val=ore_val)
                self.return_on_investment()
                RonI[i,j] = self.Return_on_Inv
                if self.inputs['Include_CO2_Trapping']==True:
                    self.CO2_capture()
                    self.profit()
                else:
                    self.profit()
                Pro[i,j] = self.Profit
   
        fig = go.Figure(data = go.Contour(x=Ton_yr_prod, y=Ore_val, z=Pro.transpose(),
                                          contours=dict(start=0,end=int(Pro.max()),size=round(Pro.max()/1e9/15)*1e9,showlabels = True),
                                          colorbar=dict(title='[USD]'),colorscale='greens'))
        #fig.add_contour(x=Ton_yr_prod, y=Ore_val, z=Pro.transpose(), contours_coloring='lines',
        #                contours=dict(start=Pro.min(),end=0,size=1e9, showlabels = True),
        #                line_width=2,showscale=False)
        fig.add_contour(x=Ton_yr_prod, y=Ore_val, z=Pro.transpose(), contours_coloring='lines',
                        line_width=2,colorscale='Electric',contours=dict(start=0,
                                                                         end=int(Pro.max())+1,
                                                                         size=int(Pro.max())+1,
                                                                         showlabels = True), 
                                                                         showscale=False)
        fig.update_layout(title = ( '<b>Profit </b><br>'+'after '+str(self.inputs['nb_yr_prod'])+ ' yrs' ),
                          xaxis_title="<b>Production [TonOre/yr]<b>",
                          yaxis_title="<b>Ore Value [USD/ton]<b>",
                          showlegend = False, plot_bgcolor='white')
        fig.update_xaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey' )
        fig.update_yaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey',
                         zerolinecolor='black',zerolinewidth=2)
        fig.show()

        # save fig object to png
        if save_pngs:
            self.fig = fig
            self.savefig_png(save_pngs+'_Profit_'+str(self.inputs['nb_yr_prod'])+'yrs', width=550, height=500)


        fig = go.Figure(data = go.Contour(x=Ton_yr_prod, y=Ore_val, z=RonI.transpose(),
                                          contours=dict(start=0,end=int(RonI.max()),size=int(RonI.max()/15),showlabels = True),
                                          colorbar=dict(title='[factor]'),colorscale='greens'))
        #fig.add_contour(x=Ton_yr_prod, y=Ore_val, z=RonI.transpose(), contours_coloring='lines',
        #                contours=dict(start=RonI.min(),end=0,size=1, showlabels = True),
        #                line_width=2,showscale=False)
        fig.add_contour(x=Ton_yr_prod, y=Ore_val, z=RonI.transpose(), contours_coloring='lines',
                        line_width=2,colorscale='Electric',contours=dict(start=0,
                                                                         end=int(RonI.max())+1,
                                                                         size=int(RonI.max())+1,
                                                                         showlabels = True), 
                                                                         showscale=False)
        fig.update_layout(title = ( '<b>Return on investment</b><br>'+'after '+str(self.inputs['nb_yr_prod'])+ ' yrs' ),
                          xaxis_title="<b>Production [TonOre/yr]<b>",
                          yaxis_title="<b>Ore Value [USD/ton]<b>",
                          showlegend = False, plot_bgcolor='white')
        fig.update_xaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey' )
        fig.update_yaxes(mirror=True,ticks='outside',showline=True,linecolor='black',gridcolor='lightgrey',
                         zerolinecolor='black',zerolinewidth=2)
        fig.show()

        # save fig object to png
        if save_pngs:
            self.fig = fig
            self.savefig_png(save_pngs+'_RonI_'+str(self.inputs['nb_yr_prod'])+'yrs', width=550, height=500)

        return Pro, RonI



    def savefig_html(self, file_name):
        self.fig.write_html(file_name+'.html')

    def savefig_png(self, file_name, width=900, height=600):
        pio.write_image(self.fig, file_name+'.png',scale=4, width=width, height=height)
