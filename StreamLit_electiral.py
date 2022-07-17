import streamlit as st
import numpy as np
import pandas as pd
import cufflinks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from colors import *
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import k_means,KMeans
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import datetime 
from sklearn.preprocessing import StandardScaler

   
columns = ['Temperature', 'Unit_Value','Date','Location','AMOUNT','User_ID']

clustring_cols = ['Temperature', 'Unit_Value','Location','AMOUNT','User_ID']

diff_in_years_columns = ['Charging_interrupt_difference_in_years','Date','Location','Amount_Average','User_ID']

diff_in_years_chart_columns = ['Charging_interrupt_difference_in_years','Amount_Average','User_ID']

clustringFeaturesTitles = ['Amount (Average)','Temperature', 'Unit Value']

clustringFeaturesMap = {"Amount (Average)": "Amount_Average", "Temperature": "Temperature", "Unit Value": "Unit_Value"}
      
trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}


   
@st.cache
def get_data():
    
    df = pd.read_csv('out.csv', low_memory=False)    

    df = df[df['AMOUNT'] != 0]

    df = df[columns]
    df["Date"] = pd.to_datetime(df.Date).dt.date
    df['Date'] = pd.DatetimeIndex(df.Date)
        
    df['Location'].replace('', np.nan, inplace=True)
    df['Location'].replace(' ', np.nan, inplace=True)
    df.dropna(subset=['Location'], inplace=True)
    df.sort_values("Date").dropna(subset=columns)
    
    return df
    
def filter_analysisData(selected, trend_level):
    
    if "All" in selected:
        trend_data_all = data.groupby([ pd.Grouper(key="Date",freq=trend_kwds[trend_level]),"User_ID"]).aggregate(Amount_Sum=("AMOUNT", "sum"),
        Temperature = ("Temperature", "mean")).reset_index() 
        
        trend_data_all = trend_data_all.groupby([ pd.Grouper(key="Date",
        freq=trend_kwds[trend_level])]).aggregate(Amount_Average=("Amount_Sum","mean"),
        Temperature = ("Temperature", "max")).reset_index()
        
        trend_data_all = trend_data_all.dropna()
    
    
    #else:
    trend_data = data.query(f"Location in {selected}").\
        groupby(["Location", pd.Grouper(key="Date",freq=trend_kwds[trend_level]),"User_ID"]).aggregate(Amount_Sum=("AMOUNT", "sum"),
        Temperature = ("Temperature", "mean")).reset_index() 
        
    trend_data = trend_data.query(f"Location in {selected}").\
        groupby(["Location", pd.Grouper(key="Date",
        freq=trend_kwds[trend_level])]).aggregate(Amount_Average=("Amount_Sum","mean"),
        Temperature = ("Temperature", "max")).reset_index()
        

    if "All" in selected:
        trend_data = pd.concat([trend_data, trend_data_all])
        
    trend_data["Date"] = trend_data.Date.dt.date
    return trend_data.round(2)
    
def filter_clustringData():

    result = data.groupby(["User_ID"]).aggregate(Amount_Sum=("AMOUNT", "sum"),
            Temperature = ("Temperature", "max"),
            Location = ('Location','first'),
            Unit_Value = ('Unit_Value','mean'),
            Date = ('Date','max') 
            ).reset_index()  


    locationResult = result.groupby(["Location","User_ID"]).aggregate(Amount_Average=("Amount_Sum", "mean"),
            Temperature = ("Temperature", "min"),
            Unit_Value = ('Unit_Value','mean'),
            Date = ('Date','max') 
            ).reset_index()  
            
    return locationResult.round(2)
    
def year_diff(d1, d2):
    later = max(d1, d2)
    earlier = min(d1, d2)

    result = later.year - earlier.year
    if later.month < earlier.month or (later.month == earlier.month and later.day < earlier.day):
        result -= 1

    return result
    
colors = get_colors()

data = get_data()

locations = data.Location.unique().tolist()

sidebar = st.sidebar

locationOptions = data["Location"].drop_duplicates()
locationOptions.loc[-1]= "All"

sidebar.title("Application mode:")

st.write('<style>div.row-widget.stRadio > label{display:none;}</style>', unsafe_allow_html=True)      
mode = sidebar.radio("", ["Analysis", "Clustering"])
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

sidebar.write('<br></br>', unsafe_allow_html=True)

st.markdown("<h1 style='color:#DC143C;'>Electrical Fraud Detection System</h1>", unsafe_allow_html=True)

 st.markdown("<h3 style='color:#F08080;'>'"+mode+"' Mode</h3>", unsafe_allow_html=True)

if mode=="Analysis": 

        with st.sidebar:
            st.subheader("Analysis Options:")
            
        selected = sidebar.multiselect("Select locations ", locationOptions, default='All')
        
        show_data = sidebar.checkbox("Show Data")

        trend_level = sidebar.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])

        trend_data = filter_analysisData(selected, trend_level)
        
        with st.sidebar:
            st.subheader("Chart Options:")
    
        Amount_Average = sidebar.checkbox("Amount (Average)")
        Temperature = sidebar.checkbox("Temperature")

        
        lines = [Amount_Average,Temperature]
        line_cols = ["Amount_Average","Temperature"]
        trends = [c[1] for c in zip(lines,line_cols) if c[0]==True]
        
        ndf = pd.DataFrame(data=trend_data.Date.unique(),columns=["Date"])

        for s in selected:
            new_cols = ["Date"]+[f"{s}_{c}" for c in line_cols]
            
            if s =="All":
                tdf = trend_data.copy()
            else:    
                tdf = trend_data.query(f"Location=='{s}'")
            
            tdf.drop("Location", axis=1, inplace=True)
            tdf.columns=new_cols
            ndf=ndf.merge(tdf,on="Date",how="inner")

        if show_data:
            if len(ndf)>0:
                AgGrid(ndf)
            else:
                st.markdown("Empty Dataframe")
        
        new_trends = []
        for c in trends:
            new_trends.extend([f"{s}_{c}" for s in selected])

        subplots=sidebar.checkbox("Show Subplots", True)
        if len(trends)>0:
            st.markdown("### Trend of Selected Locations")

            fig=ndf.iplot(kind="line", asFigure=True, xTitle="Date", yTitle="Values",
                                x="Date", y=new_trends, title=f"{trend_level} Trend of {', '.join(trends)}.", subplots=subplots)
            st.plotly_chart(fig, use_container_width=False)
            
            
if mode=="Clustering":  

    with st.sidebar:
        st.subheader("Filters:")
            
    show_clustering_data = sidebar.checkbox("Show Data")

    show_max = sidebar.checkbox("Show Max 10 charges amounts")
    show_min = sidebar.checkbox("Show Min 10 charges amounts")
    show_diff_in_year = sidebar.checkbox("Show charging interruption/difference in Years")
    
    udf = filter_clustringData()

    selectedLocation = sidebar.multiselect("Select location", locationOptions,default='All')
    
    filteredUdf = []

    if len(selectedLocation):
        if "All" in selectedLocation:
            filteredUdf = udf.copy()
            
        else:
            filteredUdf = udf[udf['Location'].isin(selectedLocation)]
    if show_clustering_data:
        if len(filteredUdf):
            st.markdown("### All values:")
            AgGrid(filteredUdf)
    
    if show_max:
        if len(filteredUdf)>0:
            st.markdown("### Max values:")
            AgGrid(filteredUdf.sort_values(by=['Amount_Average'],ascending=False).head(10))
     if show_min:
        if len(filteredUdf)>0:
            st.markdown("### Min values:")
            AgGrid(filteredUdf.sort_values(by=['Amount_Average']).head(10))
     
    if show_diff_in_year:
        if len(filteredUdf)>0:      
            ## Data
            today = datetime.date.today() 
            filteredUdfForYears = filteredUdf.copy()
            filteredUdfForYears['Charging_interrupt_difference_in_years'] = filteredUdfForYears.apply(lambda row : year_diff(today, row['Date'].date()) , axis = 1)
            filteredUdfForYears = filteredUdfForYears[diff_in_years_columns]
            
            ## Table
            st.markdown("### Difference in years for charges table:")
            AgGrid(filteredUdfForYears.sort_values(by=['Charging_interrupt_difference_in_years'],ascending=False))
           
            ## Chart
            chart_data = filteredUdfForYears[diff_in_years_chart_columns]
            chart_data.set_index('User_ID',inplace=True)
          
            
            ## Slider
            if len(chart_data)>0: 
                minValue = chart_data['Charging_interrupt_difference_in_years'].min().item()
                maxValue = chart_data['Charging_interrupt_difference_in_years'].max().item()
                 
                if minValue == maxValue:
                    minValue = minValue - 1
                    
                filterDiffInYears = sidebar.slider("Difference in years for charges", min_value=minValue, max_value=maxValue, value=minValue)
                chart_data = chart_data[chart_data['Charging_interrupt_difference_in_years'] >= filterDiffInYears]
            
            st.markdown("### Difference in years for charges chart:")
            st.bar_chart(chart_data)

    with st.sidebar:
        st.subheader("Clustering Options:")
        
    clustringFeatures = sidebar.multiselect("Select Features", clustringFeaturesTitles)#, default=columns[:1])
    
    features = []
    for f in clustringFeatures:
        features.append(clustringFeaturesMap[f])
    
    # select a clustering algorithm
    calg = sidebar.selectbox("Select a clustering algorithm", ["K-Means","K-Medoids", "Spectral Clustering", "Agglomerative Clustering"])

    # select number of clusters
    ks = sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=2)
        
    if len(features)>=2 and len(filteredUdf)>0:
        
        if calg == "K-Means":
            st.markdown("### K-Means Clustering")        
            use_pca = "No"
            if use_pca=="No":
                inertias = []
                for c in range(1,ks+1):
                    tdf = filteredUdf.copy()
                    X = tdf[features]                



                    model = KMeans(n_clusters=c)
                    model.fit(X)

                    y_kmeans = model.predict(X)
                    tdf["cluster"] = y_kmeans
                    inertias.append((c,model.inertia_))

                    trace0 = go.Scatter(x=tdf[features[0]],y=tdf[features[1]],mode='markers',  
                                        marker=dict(
                                                color=tdf.cluster.apply(lambda x: colors[x]),
                                                colorscale='Viridis',
                                                showscale=True,
                                                size = udf["Amount_Average"]%20,
                                                opacity = 0.9,
                                                reversescale = True,
                                                symbol = 'pentagon'
                                                ),
                                        name="Locations", text=udf["Location"])

                    trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                        mode='markers', 
                                        marker=dict(
                                            color=colors,
                                            size=20,
                                            symbol="circle",
                                            showscale=True,
                                            line = dict(
                                                width=1,
                                                color='rgba(102, 102, 102)'
                                                )

                                            ),
                                        name="Cluster Mean")

                    data7 = go.Data([trace0, trace1])
                    fig = go.Figure(data=data7)
                    layout = go.Layout(
                                height=600, width=800, title=f"KMeans Cluster Size {c}",
                                xaxis=dict(
                                    title=features[0],
                                ),
                                yaxis=dict(
                                    title=features[1]
                                ) ) 

                    fig.update_layout(layout)
                    st.plotly_chart(fig)

                inertias=np.array(inertias).reshape(-1,2)
                performance = go.Scatter(x=inertias[:,0], y=inertias[:,1])
                layout = go.Layout(
                    title="Cluster Number vs Inertia",
                    xaxis=dict(
                        title="Ks"
                    ),
                    yaxis=dict(
                        title="Inertia"
                    ) ) 
                fig=go.Figure(data=go.Data([performance]))
                fig.update_layout(layout)
                st.plotly_chart(fig)

            
    else:
        st.markdown("### Please Select at Least 2 Features for Visualization and 1 Location, or All")
        

