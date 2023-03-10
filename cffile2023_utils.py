import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import cryptpandas as crp


class EBITDA:

    def __init__(self, pwd, map_loc):

        # Read table data
        self.read_table(pwd, map_loc)

        # Standard input parameters
        self.inputs = {}
        self.inputs['yr_prod_vol'] = 0 
        self.inputs['nb_yr_prod']  = 0 
        self.inputs['Penalty']     = 0 
        self.inputs['RefRecovery'] = 0 
        self.inputs['ProsEx']      = 0 
        self.inputs['Processing']  = 0 
        self.inputs['RefCost']     = 0 
        self.inputs['CapEx']       = 0 
        self.inputs['ExpEnvEx']    = 0 
        self.inputs['OpEx']        = 0 
        self.inputs['selected_location'] = 7 
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

    def return_on_investment(self):
        return_ = ( self.Net_ore_value_sum 
                      - self.Penalty_sum
                      - self.RefLoss_sum 
                      - self.RefCost_sum
                      - self.ProsEx_sum
                      - self.OpEx_sum )
        investment_ =  self.inputs['CapEx'] + self.inputs['ExpEnvEx']
        self.Return_on_Inv = (return_ - investment_)/investment_
    

    def variable_sums(self, nb_yr_prod):

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
        self.Elements_value = self.selected_elements_USD_Ton * self.total_vol
        self.Net_ore_value_sum = np.sum(self.Elements_value) 
        self.Penalty_sum = (self.inputs['Penalty'] * self.total_vol)
        self.RefLoss_sum = (self.Net_ore_value_sum-self.Penalty_sum) * (1-self.inputs['RefRecovery'])
        self.RefCost_sum = self.inputs['Processing'] * (self.inputs['RefCost']*self.total_vol)
        self.ProsEx_sum = self.inputs['ProsEx'] * self.total_vol
        self.OpEx_sum = self.inputs['OpEx'] * nb_yr_prod


    def waterfall(self):

        # Call functions
        #self.selected_elements()
        self.variable_sums(nb_yr_prod = self.inputs['nb_yr_prod'])
        self.profit()

        # Make waterfall plotly plot
        measure = list(np.repeat('relative', len(self.inputs['selected_elements']) + 7)) + ['total']
        x = self.inputs['selected_elements'] + ['Penalty', 'RefLoss', 'RefCost', 'ProsEx', 'CapEx', 'ExpEnvEx', 'OpEx', 'Profit']
        y = ( list(self.Elements_value) 
            + list([-self.Penalty_sum, -self.RefLoss_sum, -self.RefCost_sum, -self.ProsEx_sum, -self.inputs['CapEx'], -self.inputs['ExpEnvEx'], -self.OpEx_sum, self.Profit]) )

        y = np.array(y)/1e6 # Plot in mUSD

        color_profit = "#2E8B57" # green if positive
        if self.Profit<0:
            color_profit = "#FF2400" # red if negative

        fig = go.Figure(go.Waterfall(
            measure = measure,
            x = x,
            textposition = "outside",
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


    def savefig_html(self, file_name):
        self.fig.write_html(file_name+'.html')


