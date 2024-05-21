import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr


# set to wide mode
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")


do_stuff_on_page_load()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load catchment list
catchments = pd.read_csv("./data/Caravan-CAMELS/catchments.csv", dtype=str)

# load optimal parameters
optimal_paras = np.genfromtxt("./data/camels_embeddings.csv", delimiter=",")

# load model
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

decoder.to(device=device)

decoder.eval()

# Input time series
st.sidebar.markdown(
    "## Select a catchment from the CAMELS dataset or upload a climate forcing and discharge time series."
)

selected_catchment = st.sidebar.selectbox(
    "Which CAMELS catchment do you want to model?",
    catchments["gauge_name"].tolist(),
    index=0,
    placeholder="Choose a catchment...",
)

uploaded_file = st.sidebar.file_uploader(
    "Or, upload a comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file, delimiter=",", header=None)
    input_data = input_data.to_numpy()
    x = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32).to(device=device)
    optimal_para=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
else:
    index = catchments[catchments["gauge_name"] == selected_catchment].index.tolist()
    optimal_para = optimal_paras[index[0],:].tolist()
    file_name = catchments[catchments["gauge_name"] == selected_catchment][
        "data_all"
    ].to_string(index=False)
    input_data = np.genfromtxt(file_name, delimiter=",")
    x = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32).to(device=device)


# input parameter values
st.sidebar.markdown(
    "## Select an 8-dimensional numerical vector (i.e., the eight parameter values) below to generate a model instance."
)
number1 = st.sidebar.slider(
    "Select value of parameter 1",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[0],
)
number2 = st.sidebar.slider(
    "Select value of parameter 2",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[1],
)
number3 = st.sidebar.slider(
    "Select value of parameter 3",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[2],
)
number4 = st.sidebar.slider(
    "Select value of parameter 4",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[3],
)
number5 = st.sidebar.slider(
    "Select value of parameter 5",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[4],
)
number6 = st.sidebar.slider(
    "Select value of parameter 6",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[5],
)
number7 = st.sidebar.slider(
    "Select value of parameter 7",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[6],
)
number8 = st.sidebar.slider(
    "Select value of parameter 8",
    min_value=-11.0,
    max_value=11.0,
    value=optimal_para[7],
)


# input warm-up period
st.sidebar.markdown("## Select a warm-up period.")

warm_up = st.sidebar.number_input(
    "How many days of input are used for warm up?",
    min_value=0,
    max_value=x.shape[1],
    value=min(365, x.shape[1]),
)

# prediction
solution = np.array(
    [number1, number2, number3, number4, number5, number6, number7, number8]
)

solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32).to(device=device)
solution = solution.expand(x.shape[0], -1)

pred = decoder.decode(solution, x, base_length=warm_up).view(-1).detach().cpu().numpy()

# process prediction result
d = {
    "Simulated [mm/day]": pred.tolist(),
    "Observation [mm/day]": input_data[warm_up:, 3].tolist(),
}

chart_data = pd.DataFrame(data=d)

# Plotting
st.header("Generative Hydrological Modeling")
st.subheader(
    "Generating hydrological model instances from numerical vectors in the latent space for hydrological prediction."
)

st.markdown(
    "*Use the sidebar to specify the numeric vector values and data sources to generate model instances and run simulations.*"
)


if uploaded_file is not None:
    st.markdown("User supplied catchment data")
else:
    st.markdown(
        f"Comparison of simulated and observed hydrographs of the :red[{selected_catchment}, USA]."
    )

dispaly_days = st.slider(
    "*Select a period to display:*",
    0,
    x.shape[1] - warm_up,
    (0, min(1000, x.shape[1] - warm_up)),
)

st.line_chart(
    chart_data[dispaly_days[0] : dispaly_days[1]], color=["#0457ac", "#a7e237"]
)

if uploaded_file is None:
    st.caption(
        "Simulation starts from 1981-01-01, and the warm-up period is not shown. Data are from the Caravan dataset."
    )


# Prediction accuracy

kge = HydroErr.kge_2009(simulated_array=pred, observed_array=input_data[warm_up:, 3])
kge = round(kge, 3)

nse = HydroErr.nse(simulated_array=pred, observed_array=input_data[warm_up:, 3])
nse = round(nse, 3)

f"Performance of the generated model instance on all data: :red[**KGE={kge}**], :red[**NSE={nse}**]."


# Prediction accuracy of displayed period
kge2 = HydroErr.kge_2009(
    simulated_array=pred[dispaly_days[0] : dispaly_days[1]],
    observed_array=input_data[
        (dispaly_days[0] + warm_up) : (dispaly_days[1] + warm_up), 3
    ],
)
kge2 = round(kge2, 3)

nse2 = HydroErr.nse(
    simulated_array=pred[dispaly_days[0] : dispaly_days[1]],
    observed_array=input_data[
        (dispaly_days[0] + warm_up) : (dispaly_days[1] + warm_up), 3
    ],
)
nse2 = round(nse2, 3)

f"Performance of the generated model instance in the displayed period: :red[**KGE={kge2}**], :red[**NSE={nse2}**]."

# Display data
st.divider()

st.write("The simulated and observed discharge data without the warm-up period:")
st.write(chart_data)

# References:
st.divider()

st.markdown(
    "The method for developing generative hydrological model is discribed in the paper: [Learning to Generate Lumped Hydrological Models](https://arxiv.org/abs/2309.09904)."
)
st.caption(
    "The CAMELS catchment data was derived from the [Caravan dataset](https://doi.org/10.1038/s41597-023-01975-w). License of the dataset can be found in the GitHub page of this web appplication."
)

st.markdown(
    '<a href="mailto:yyang90@connect.hku.hk">Contact the authors. </a>',
    unsafe_allow_html=True,
)
