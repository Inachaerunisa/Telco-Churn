import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
from sklearn.preprocessing import FunctionTransformer

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service' : 'No'})
df['TotalCharges'] = df['TotalCharges'].replace(' ',0)
df['TotalCharges'] = df['TotalCharges'].astype(float)
gabung = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in gabung : 
    df[i]  = df[i].replace({'No internet service' : 'No'})
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'No' if x==0 else
                                                           'Yes' if x==1 else x)
    
app.layout = html.Div(children = [
    dcc.Tabs(value = 'tabs', id='tabs-1', children = [
        dcc.Tab(label='Home', className='col-6', children = 
        html.Div(children=[
            html.Div(children = [
                html.Center(html.H1('Telco Customer Churn')),
                html.Center(html.H2('by: Ina',style = {'font-weight' : 'bold', 'font-style' : 'italic'})),
                html.Div(html.Img(src = '/assets/churn.jpg'),
                    style ={'text-align' : 'center', 'margin-top' : '80px',
                            'margin-bottom' : '80px'}),

            html.Div(id = 'tabel',
            children = [dash_table.DataTable(id='data_table',
            columns=[{"name": i, 'id' : i} for i in df.columns],
            data = df.to_dict('records'),
            sort_action='native',
            filter_action='native',
            page_action = 'native',
            page_current = 0,
            page_size = 10,
            style_table = {'overflowX': 'scroll'})])
            ])
        ] )),
        dcc.Tab(label='Telco Customer Prediction', children = [
            html.Div(children=[
                html.Div(children=[html.P('Tenure'),
                html.Div(children=[dcc.Input(
                    id='tenure',type='number',min=0,max=100,step=1
                )],className='col-5')],className='col-5'
                        ),

                html.Div(children=[html.P('Senior Citizen'),
                html.Div(children=[dcc.Dropdown(
                    id='senior_citizen',options=[{'label': i, 'value':i} for i in df['SeniorCitizen'].unique()]
                )],className='col-5')],className='col-5'
                        )],className='row'),

            html.Div(children=[
                html.Div(children=[html.P('Phone Service'),
                html.Div(children=[dcc.Dropdown(
                    id='phone_service',options=[{'label': i, 'value':i} for i in df['PhoneService'].unique()]
                )],className='col-5')],className='col-5'
                        ),

                html.Div(children=[html.P('Multiple Lines'),
                html.Div(children=[dcc.Dropdown(
                    id='multiplelines',options=[{'label': i, 'value':i} for i in df['MultipleLines'].unique()]
                )],className='col-5')],className='col-5'
                        )],className='row'),
                        
            html.Div(children=[
                html.Div(children=[html.P('InternetService'),
                html.Div(children=[dcc.Dropdown(
                    id='internet_service',options=[{'label': i, 'value':i} for i in df['InternetService'].unique()]
                )],className='col-5')],className='col-5'
                        ),

                html.Div(children=[html.P('Online Security'),
                html.Div(children=[dcc.Dropdown(
                    id='online_security',options=[{'label': i, 'value':i} for i in df['OnlineSecurity'].unique()]
                )],className='col-5')],className='col-5'
                        )],className='row'),      
                        

            html.Div(children=[
                html.Div(children=[html.P('Tech Support'),
                html.Div(children=[dcc.Dropdown(
                    id='tech_support',options=[{'label': i, 'value':i} for i in df['TechSupport'].unique()]
                )],className='col-5')],className='col-5'
                        ),

                html.Div(children=[html.P('Contract'),
                html.Div(children=[dcc.Dropdown(
                    id='contract',options=[{'label': i, 'value':i} for i in df['Contract'].unique()]
                )],className='col-5')],className='col-5'
                        )],className='row'), 

            html.Div(children=[
                html.Div(children=[html.P('Payment Method'),
                html.Div(children=[dcc.Dropdown(
                    id='payment_method',options=[{'label': i, 'value':i} for i in df['PaymentMethod'].unique()]
                )],className='col-5')],className='col-5'
                        ),

                html.Div(children=[html.P('Total Charges'),
                html.Div(children=[dcc.Input(
                    id='total_charges',type='number',min=1,max=1000000
                )],className='col-5')],className='col-5'
                        )],className='row'),    

            html.Div(children=[html.P('Paperless Billing'),
                html.Div(children=[dcc.Dropdown(
                    id='paperlessbilling',options=[{'label': i, 'value':i} for i in df['PaperlessBilling'].unique()]
                )],className='col-5')],className='col-5'
                        ),
            html.Div(html.Button('Search', id = 'search'),className='col-3'),
            html.Div(children=[html.Center(html.H1('Hasil Prediksi'))],id='hasil')        

                        ])
    ])
])


@app.callback(
    Output(component_id='hasil',component_property='children'),
    [Input(component_id='search',component_property='n_clicks')],
    [State(component_id='tenure',component_property='value'),
    State(component_id='senior_citizen',component_property='value'),
    State(component_id='phone_service',component_property='value'),
    State(component_id='multiplelines',component_property='value'),
    State(component_id='internet_service',component_property='value'),
    State(component_id='online_security',component_property='value'),
    State(component_id='tech_support',component_property='value'),
    State(component_id='contract',component_property='value'),
    State(component_id='payment_method',component_property='value'),
    State(component_id='total_charges',component_property='value'),
    State(component_id='paperlessbilling',component_property='value')
    ])

def check(n_clicks,tenure_,seniorcitizen_,phoneservice_,multiplelines_,internetservice_,onlinesecurity_,techsupport_,contract_,paymentmethod_,totalcharges_,paperlessbilling_):
    contract_One = 0
    contract_Two = 0
    internetservice_Fiber = 0
    internetservice_No = 0

    if n_clicks ==None:
        return 'Hasil Prediksi'
    else:
        if seniorcitizen_ == 'Yes':
            seniorcitizen_ = 1
        else:
            seniorcitizen_ = 0

        if phoneservice_ == 'Yes':
            phoneservice_ = 1
        else:
            phoneservice_ = 0

        if multiplelines_ == 'Yes':
            multiplelines_ = 1
        else:
            multiplelines_ = 0

        if internetservice_ == 'Fiber optic':
            internetservice_Fiber = 1
            internetservice_No = 0
        elif internetservice_ == 'No':
            internetservice_Fiber = 0
            internetservice_No = 1
        else:
            internetservice_Fiber = 0
            internetservice_No = 0

        if onlinesecurity_ == 'Yes':
            onlinesecurity_ = 1
        else:
            onlinesecurity_ = 0
            
        if techsupport_ == 'Yes':
            techsupport_ = 1
        else:
            techsupport_ = 0

        if contract_ == 'One year':
            contract_One = 1
            contract_Two = 0
        elif contract_ == 'Two year':
            contract_One = 0
            contract_Two = 1
        else:
            contract_One = 0
            contract_Two = 0

        if paperlessbilling_ == 'Yes':
            paperlessbilling_ = 1
        else:
            paperlessbilling_ = 0

        if paymentmethod_ == 'Electronic check':
            paymentmethod_ = 1
        else:
            paymentmethod_ = 0

        model = pickle.load(open('logistik.sav', 'rb'))
        predict = model.predict(np.array([tenure_,totalcharges_,seniorcitizen_,phoneservice_,multiplelines_,internetservice_Fiber,internetservice_No,onlinesecurity_,techsupport_,contract_One,contract_Two,paperlessbilling_,paymentmethod_]).reshape(1,-1))[0]
        proba = model.predict_proba(np.array([tenure_,totalcharges_,seniorcitizen_,phoneservice_,multiplelines_,internetservice_Fiber,internetservice_No,onlinesecurity_,techsupport_,contract_One,contract_Two,paperlessbilling_,paymentmethod_]).reshape(1,-1))[0][predict]
        if predict == 0 :
           return html.Center(html.H1('No Churn {}'.format(round(proba, 2))))
        else :
            return html.Center(html.H1('Churn {}'.format(round(proba, 2))))
        

if __name__ == '__main__':
    app.run_server(debug=True)
