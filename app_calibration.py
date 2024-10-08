import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr

import pygad

import random


# set to wide mode
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")


do_stuff_on_page_load()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load catchment list
catchments = pd.read_csv("./data/Caravan-CAMELS/catchments.csv", dtype=str)

# write title texts
st.header("Generative hydrological modeling")

st.subheader(
    "Hydrological model calibration: search the 8-dimensional latent space to identify the optimal numerical vectors for defining discharge prediction model."
)

st.markdown(
    "*Use the sidebar to specify data sources and genetic algorithm (GA) parameter values.*"
)


st.markdown(
    'Click the :blue["Run optimization"] button to see calibration results, simulated hydrographs, predictions, etc.'
)

# load model
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

decoder.to(device=device)

decoder.eval()

# Input calibration time series

st.sidebar.markdown(
    "## Select a catchment from the Caravan dataset or upload a climate forcing and discharge time series."
)

selected_catchment = st.sidebar.selectbox(
    "Which CAMELS catchment do you want to model?",
    catchments["gauge_name"].tolist(),
    index=0,
    placeholder="Choose a catchment...",
)

st.sidebar.markdown(
    "## Or, upload *calibration* and *test* climate forcing and discharge time series data."
)

uploaded_file_calibration = st.sidebar.file_uploader(
    "Select a *calibration* period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)

uploaded_file_test = st.sidebar.file_uploader(
    "Select a *test* period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)


if uploaded_file_calibration is not None:
    input_data = pd.read_csv(uploaded_file_calibration, delimiter=",", header=None)
    input_data = input_data.to_numpy()
    x_cal = (
        torch.from_numpy(input_data[:, 0:3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
    y_cal = (
        torch.from_numpy(input_data[:, 3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
else:
    file_name = catchments[catchments["gauge_name"] == selected_catchment][
        "data_train"
    ].to_string(index=False)
    input_data = np.genfromtxt(file_name, delimiter=",")
    x_cal = (
        torch.from_numpy(input_data[:, 0:3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
    y_cal = (
        torch.from_numpy(input_data[:, 3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )


if uploaded_file_test is not None:
    input_data = pd.read_csv(uploaded_file_test, delimiter=",", header=None)
    input_data = input_data.to_numpy()
    x_test = (
        torch.from_numpy(input_data[:, 0:3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
    y_test = (
        torch.from_numpy(input_data[:, 3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
else:
    file_name = catchments[catchments["gauge_name"] == selected_catchment][
        "data_test"
    ].to_string(index=False)
    input_data = np.genfromtxt(file_name, delimiter=",")
    x_test = (
        torch.from_numpy(input_data[:, 0:3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
    y_test = (
        torch.from_numpy(input_data[:, 3])
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )


# input warm-up period
st.sidebar.markdown("## Select a warm-up period.")

warm_up = st.sidebar.number_input(
    "How many days of input are used for warm up?",
    min_value=0,
    max_value=x_cal.shape[1],
    value=min(365, x_cal.shape[1]),
)

# input a random seed of GA
st.sidebar.markdown("## Select the random seed of generations of GA.")

random_seed = st.sidebar.number_input(
    "Input a random seed.",
    min_value=0,
    max_value= 4294967295,
    value=random.randint(0, 4294967295),
)

# input number of generaion
st.sidebar.markdown("## Select the number of generations of GA.")

num_generations = st.sidebar.number_input(
    "Input the number of generations.",
    min_value=2,
    max_value=500,
    value=42,
)

# input sol_per_pop
st.sidebar.markdown("## Select the population size of GA.")

sol_per_pop = st.sidebar.number_input(
    "Input the population size.",
    min_value=20,
    max_value=500,
    value=28,
)


# input sol_per_pop
st.sidebar.markdown("## Select the number of solutions to be selected as parents.")

num_parents_mating = st.sidebar.number_input(
    "Input the number of solutions to be selected as parents.",
    min_value=2,
    max_value=sol_per_pop,
    value=16,
)


class Objective_builder:
    def __init__(self, x, y):
        self.x = x.contiguous()
        self.y = y.contiguous()

    def eval(self, ga_instance, solution, solution_idx):
        # numpy to torch tensor
        solution = (
            torch.from_numpy(solution)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .to(device=device)
        )
        solution = solution.expand(self.x.shape[0], -1)

        # BASE_LENGTH is from global
        pred = (
            decoder.decode(solution, self.x, base_length=warm_up)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )

        ob = self.y.view(-1).detach().cpu().numpy()[warm_up:]

        return HydroErr.kge_2009(simulated_array=pred, observed_array=ob)

    def pred(self, solution):
        # numpy to torch tensor
        solution = (
            torch.from_numpy(solution)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .to(device=device)
        )
        solution = solution.expand(self.x.shape[0], -1)

        # BASE_LENGTH is from global
        pred = (
            decoder.decode(solution, self.x, base_length=warm_up)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )

        ob = self.y.view(-1).detach().cpu().numpy()[warm_up:]

        d = {
            "Simulated [mm/day]": pred.tolist(),
            "Observation [mm/day]": ob.tolist(),
        }

        chart_data = pd.DataFrame(data=d)

        return chart_data


# Hyperparameters of GA
num_genes = 8

init_range_low = -11
init_range_high = 11

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_probability = 0.25

fn_cal = Objective_builder(x_cal, y_cal)
fn_test = Objective_builder(x_test, y_test)

# run optimization
if "clicked" not in st.session_state:
    st.session_state.clicked = False

progress_text = (
    "Optimization in progress. Please wait. It may take from 1 to several minutes."
)

if st.session_state.clicked:
    my_bar = st.progress(0, text=progress_text)


def on_generation(instance):
    generations_completed = instance.generations_completed

    my_bar.progress(generations_completed / num_generations, text=progress_text)


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fn_cal.eval,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    random_seed = random_seed,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_probability=mutation_probability,
    on_generation=on_generation,
)


def click_button():
    st.session_state.clicked = True


st.button(":blue[Run optimization]", on_click=click_button)

if st.session_state.clicked:

    # run simulation
    ga_instance.run()

    chart_cal = fn_cal.pred(ga_instance.best_solution()[0])
    chart_test = fn_test.pred(ga_instance.best_solution()[0])

    # gof
    kge_cal = round(
        HydroErr.kge_2009(
            simulated_array=chart_cal["Simulated [mm/day]"],
            observed_array=chart_cal["Observation [mm/day]"],
        ),
        3,
    )  # round(ga_instance.best_solution()[1], 3)
    kge_test = round(
        HydroErr.kge_2009(
            simulated_array=chart_test["Simulated [mm/day]"],
            observed_array=chart_test["Observation [mm/day]"],
        ),
        3,
    )  # round(fn_test.eval(0, ga_instance.best_solution()[0], 0), 3)

    nse_cal = round(
        HydroErr.nse(
            simulated_array=chart_cal["Simulated [mm/day]"],
            observed_array=chart_cal["Observation [mm/day]"],
        ),
        3,
    )  # round(ga_instance.best_solution()[1], 3)
    nse_test = round(
        HydroErr.nse(
            simulated_array=chart_test["Simulated [mm/day]"],
            observed_array=chart_test["Observation [mm/day]"],
        ),
        3,
    )  # round(fn_test.eval(0, ga_instance.best_solution()[0], 0), 3)

    # session information
    if uploaded_file_calibration is not None:
        st.markdown("Calibration results of user supplied catchment data.")
    else:
        st.markdown(
            f"Performance of generative model with optimized latent variable values: :red[**Calibration KGE={kge_cal}**], :red[**Test KGE={kge_test}**];  :red[**Calibration NSE={nse_cal}**], :red[**Test NSE={nse_test}**]."
        )
        st.markdown(
            f"Calibration results of the :red[{selected_catchment}, USA]. Calibration period: 1981-01-01 to 1995-12-31; Test period: 1996-01-01 to 2014-12-31."
        )

    st.divider()

    # Optimal model parameters
    st.markdown("### Optimal numerical vector in the latent space:")

    optimal_para = pd.DataFrame(ga_instance.best_solution()[0])
    optimal_para["Latent spae dimension/parameter no."] = range(
        1, len(optimal_para) + 1
    )

    optimal_para.columns = ["Optimal value", "Latent spae dimension/parameter no."]
    st.dataframe(
        optimal_para[["Latent spae dimension/parameter no.", "Optimal value"]],
        hide_index=True,
    )

    st.divider()

    # Show test result:

    st.markdown(
        "### Comparison of simulated and observed hydrographs of the *test period* (without the warm-up period):"
    )

    st.line_chart(chart_test, color=["#0457ac", "#a7e237"])
    f":red[**Test KGE={kge_test}**], :red[**Test NSE={nse_test}**]."

    st.write(
        "### The *test period* simulated and observed discharge data (without the warm-up period):"
    )
    st.dataframe(chart_test)

    st.divider()

    # Show calibration result:
    st.markdown(
        "### Comparison of simulated and observed hydrographs of the *calibration period:* (without the warm-up period):"
    )

    st.line_chart(chart_cal, color=["#0457ac", "#a7e237"])

    f":red[**Calibration KGE={kge_cal}**], :red[**Calibration NSE={nse_cal}**]."

    st.write(
        "### The *calibration period* simulated and observed discharge data (without the warm-up period):"
    )
    st.dataframe(chart_cal)

    st.session_state.clicked = False


# References:
st.divider()

st.markdown(
    "The method for developing generative hydrological model is discribed in the paper: [Learning to Generate Lumped Hydrological Models](https://arxiv.org/abs/2309.09904)."
)
st.caption(
    "The catchment data was derived from the [Caravan dataset](https://doi.org/10.1038/s41597-023-01975-w). License of the dataset can be found in the GitHub page of this web appplication."
)

st.markdown(
    '<a href="mailto:yyang90@connect.hku.hk">Contact the authors. </a>',
    unsafe_allow_html=True,
)
