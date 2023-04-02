"""
   Copyright 2022 Marcus D. Liebenthal

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import copy
import time
import numpy as np

def startClient(suave):
    """
    This function starts the client for the SuaveStateScanner to visualize and interact with the results
    :param suave: the SuaveStateScanner object
    """
    from dash import Dash, dcc, html, Input, Output, no_update
    from dash.exceptions import PreventUpdate
    import plotly.graph_objects as go
    app = Dash(__name__)  # create dash app
    # get label of y-axis
    if suave.printVar == 0:
        data0 = suave.E
    elif suave.printVar < 0:
        data0 = suave.P[:, :, suave.printVar]
    else:
        data0 = suave.P[:, :, suave.printVar - 1]
    minE = np.nanmin(suave.E)
    maxE = np.nanmax(suave.E)
    app.layout = html.Div([
        html.Div([  # create a div for the plot
            dcc.Graph(id='graph',
                      figure={
                          'data': [go.Scatter(x=suave.allPnts, y=data0[state, :], mode='lines+markers',
                                              name="State {}".format(state)) for state in range(suave.numStates)],
                          'layout': go.Layout(title="SuaveStateScanner", xaxis={'title': 'Reaction Coordinate'},
                                              hovermode='closest',
                                              uirevision='constant',
                                              # set dark theme
                                              paper_bgcolor='rgba(42,42,42,0.42)',
                                              plot_bgcolor='rgba(42,42,42,0.80)',
                                              font={'color': 'white'})

                      },
                      style={'height': 800},
                      config={'displayModeBar': False}
                      ),
        ]),
        html.Div([  # create a div for the buttons

            html.H2("Controls", style={'text-align': 'center'}),

            html.Div([  # Print Selection
                html.Label("Print Selection"),
                dcc.Dropdown(id="print-var", value=0,
                             options=[{'label': "Energy" if i == 0 else "Property " + str(i), 'value': i}
                                      for i
                                      in range(suave.numProps + 1)], clearable=False),
                html.Div([  # Redraw, Sweep and Reorder, Abort
                    html.Button('Redraw', id='redraw', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '100%'}),
                    html.Button('Sweep Backwards', id='button-backwards', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '50%'}),
                    html.Button('Sweep Forwards', id='button-forwards', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '50%'}),
                    html.Button('Alternate Sweep', id='button-alternate', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '100%'}),
                    html.Button('Abort', id='stop-button', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '100%'}),
                    html.Button('Reset', id='reset-button', n_clicks=0,
                                style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                       'font-size': '14px', 'width': '100%'}),
                ], style={'display': 'inline-block'}),
            ], style={'display': 'inline-block', 'padding': '10px'}),
            html.Div([  # Save
                html.Button('Save Output', id='save-button', n_clicks=0,
                            style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                   'font-size': '14px', 'width': '20%'}),
            ], style={'display': 'inline-block', 'width': '100%'}),
            html.Div([  # Undo Redo
                html.Button('Undo', id='undo', n_clicks=0,
                            style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                   'font-size': '14px', 'width': '10%'}),
                html.Button('Redo', id='redo', n_clicks=0,
                            style={'padding': '10px', 'background-color': '#1E90FF', 'color': '#222222',
                                   'font-size': '14px', 'width': '10%'}),
            ], style={'display': 'inline-block', 'width': '100%'}),
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([  # create a div for swapping two states by index with button
            html.Div([
                html.Button('Swap States', id='swap-button', n_clicks=0,
                            style={'background-color': '#1E90FF', 'color': '#222222', 'font-size': '14px',
                                   'padding': '10px', 'width': '100%'}),  # make a button to start the animation
            ], style={'display': 'inline-block', 'padding': '2px 10px 2px 2px', 'margin': 'auto'}),
            html.Div([
                html.Label('State 1:'),  # make an input for state 1
                dcc.Input(id='swap-input1', type='number', value=0, min=0, max=suave.numStates - 1, step=1,
                          style={'background-color': '#FFFFFF', 'color': '#222222', 'font-size': '14px',
                                 'padding': '10px', 'width': '100%'}),
            ], style={'display': 'inline-block', 'padding': '2px 2px 2px 2px', 'margin': 'auto'}),
            html.Div([
                html.Div([
                    html.Label('State 2:'),  # make an input for state 2
                    dcc.Input(id='swap-input2', type='number', value=1, min=0, max=suave.numStates - 1, step=1,
                              style={'background-color': '#FFFFFF', 'color': '#222222', 'font-size': '14px',
                                     'padding': '10px', 'width': '100%'}),
                ], style={'display': 'inline-block', 'padding': '2px 2px 2px 2px', 'margin': 'auto'}),
                html.Div([  # create a div for shuffle
                    html.Button('Shuffle Values', id='shuffle-button', n_clicks=0,
                                style={'background-color': '#1E90FF', 'color': '#222222', 'font-size': '14px',
                                       'padding': '10px', 'width': '100%'}),
                    # make a button to start the animation
                ], style={'display': 'inline-block', 'padding': '2px 2px 2px 50px', 'margin': 'auto'}),
            ], style={'display': 'inline-block', 'padding': '2px 2px 2px 50px', 'margin': 'auto'}),
        ], style={'display': 'inline-block', 'width': '100%'}),

        html.Div([  # create a div for removing state at point
            html.Div([
                html.Button('Remove State Info', id='remove-button', n_clicks=0,
                            style={'background-color': '#1E90FF', 'color': '#222222', 'font-size': '14px',
                                   'padding': '10px', 'width': '100%'}),  # make a button to start the animation
            ], style={'display': 'inline-block', 'padding': '2px 10px 2px 2px', 'margin': 'auto'}),
            html.Div([
                html.Label('State:'),  # make an input for state to remove
                dcc.Input(id='remove-state-input', type='number', value=0, min=0, max=suave.numStates - 1, step=1,
                          style={'background-color': '#FFFFFF', 'color': '#222222', 'font-size': '14px',
                                 'padding': '10px', 'width': '100%'}),
            ], style={'display': 'inline-block', 'padding': '2px 2px 2px 2px', 'margin': 'auto'}),
        ], style={'display': 'inline-block', 'width': '100%'}),

        dcc.Loading(id="loading-save", children=[html.Div(id="loading-save-out")], type="default"),
        dcc.Loading(id="loading-reorder", children=[html.Div(id="loading-reorder-out")], type="default"),

        html.H2("Settings", style={'text-align': 'center'}),

        html.Div([  # Property List
            html.Label("Property List"),
            dcc.Checklist(id="prop-list",
                          options=[{'label': str(i + 1), 'value': i} for i in range(suave.numProps)],
                          value=suave.propList, labelStyle={'display': 'inline-block', 'padding': '10px'},
                          inputStyle={'background-color': '#1E90FF', 'color': '#222222', 'font-size': '14px'}),
            html.Div([  # make a div for all checklist options

                html.Div([  # create a div for checklist interpolate missing values
                    dcc.Checklist(id='interpolate',
                                  options=[{'label': 'Interpolative Reorder', 'value': 'interpolate'}],
                                  value=False),
                ], style={'display': 'inline-block', 'width': '33%'}),

                html.Div([  # create a div for checklist redundant swaps
                    dcc.Checklist(id='redundant', options=[{'label': 'Redundant Swaps', 'value': 'redundant'}],
                                  value=True),
                ], style={'display': 'inline-block', 'width': '33%'}),

            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0px 10px 0px 10px', 'margin': 'left'}),

            html.Div([  # make a div for all manual inputs
                html.Div([  # create a div for number of sweep to do
                    html.Label("Number of Sweeps:"),
                    dcc.Input(id='numSweeps', type='number', value=1, min=1, max=100),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for stencil width
                    html.Label("Stencil Width:"),
                    dcc.Input(placeholder="Stencil Width", id="stencil-width", type="number", value=suave.width,
                              debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for maxPan
                    html.Label("Max Pan of Stencil:"),
                    dcc.Input(placeholder="Max Pan", id="max-pan", type="number", value=1000, debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for derivative order
                    html.Label("Derivative Order:"),
                    dcc.Input(placeholder="Derivative Order", id="order-value", type="number", value=suave.orders[0],
                              debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),
            ], style={'width': '65%', 'display': 'inline-block', 'padding': '0px 10px 0px 10px',
                      'margin': 'right'}),
        ], style={'display': 'inline-block', 'width': '100%', 'padding': '10px'}),

        html.Div([  # make a slider to control the points to be reordered
            html.Div("Point Range",
                     style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
            dcc.RangeSlider(id="point-slider", min=0, max=suave.numPoints, step=1, value=suave.pntBounds,
                            marks={i: "{}".format(suave.allPnts[i]) for i in range(suave.numPoints)},
                            allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

        html.Div([  # make a slider to control the states to be reordered
            html.Div("State Range",
                     style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
            dcc.RangeSlider(id="state-slider", min=0, max=suave.numStates, step=1, value=suave.stateBounds,
                            marks={i: "{}".format(i) for i in range(suave.numStates)},
                            allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

        html.Div([  # make a slider to control the energy range
            html.Div("Energy Range",
                     style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
            dcc.RangeSlider(id="energy-slider", min=minE - 1e-3, max=maxE + 1e-3, step=1e-6,
                            value=[minE - 1e-3, maxE + 1e-3],
                            marks={minE: "Minimum Energy", maxE: "Maximum Energy"},
                            allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

        html.Div([  # create a div for energy width
            html.Div("Energy Width",
                     style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
            dcc.Slider(id="energy-width", value=abs(maxE - minE) + 1e-3, min=0, max=abs(maxE - minE) + 1e-3,
                       marks={1e-12: "Minimum Energy Width", abs(maxE - minE) + 1e-3: "Maximum Energy Width"},
                       step=1e-6, tooltip={'always_visible': True, 'placement': 'left'}),
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 20px 50px', 'margin': 'auto'}),
    ], style={'width': '100%', 'padding': '10px 10px 10px 10px', 'margin': 'auto',
              # set dark theme
              'backgroundColor': '#111111', 'color': '#7FDBFF'})
    sweep = 0  # initialize sweep counter

    def make_figure():
        """
        This function creates the figure to be plotted
        :return fig: the figure to be plotted
        """
        nonlocal sweep

        # update plot data
        lastEvals, lastPvals = suave.prepareSweep()  # save last energies and properties
        if suave.printVar == 0:
            data = suave.E
        elif suave.printVar < 0:
            data = suave.P[:, :, suave.printVar]
        else:
            data = suave.P[:, :, suave.printVar - 1]

        # create figure
        fig = go.Figure(
            data=[go.Scatter(x=suave.allPnts, y=data[state, :], mode='lines+markers', name="State {}".format(state))
                  for state in range(suave.numStates)],

            layout=go.Layout(title="SuaveStateScanner", xaxis={'title': 'Reaction Coordinate'},
                             hovermode='closest',
                             uirevision='constant',
                             # set dark theme
                             paper_bgcolor='rgba(42,42,42,0.42)',
                             plot_bgcolor='rgba(42,42,42,0.80)',
                             font={'color': 'white'})
        )

        suave.E = copy.deepcopy(lastEvals)
        suave.P = copy.deepcopy(lastPvals)
        return fig, f"Sweep {sweep}"

    eval_undo = []  # initialize history of energies for undo
    pval_undo = []  # initialize history of properties for undo
    eval_redo = []  # initialize memory of energies for redo
    pval_redo = []  # initialize memory of properties for redo
    original_eval = copy.deepcopy(suave.E)  # save the original energies
    original_pval = copy.deepcopy(suave.P)  # save the original properties

    def undo_callback():
        """
        This function undoes the last sweep
        :return Boolean: True if successful, False otherwise
        """
        nonlocal eval_undo, pval_undo, eval_redo, pval_redo
        if len(eval_undo) < 1:  # if there is nothing to undo
            return False

        if len(eval_redo) > 10:  # if there are too many redos
            # remove the oldest redo
            eval_redo.pop(0)
            pval_redo.pop(0)

        # add the current state to the redo memory
        eval_redo.append(suave.E)
        pval_redo.append(suave.P)

        # set the current state to the previous state and remove the previous state from the undo memory
        suave.E = copy.deepcopy(eval_undo.pop(-1))
        suave.P = copy.deepcopy(pval_undo.pop(-1))

        return True

    def redo_callback():
        """
        This function redoes the last sweep
        :return None
        """
        nonlocal eval_undo, pval_undo, eval_redo, pval_redo
        if len(eval_redo) < 1:  # if there is nothing to redo
            return False  # do nothing

        if len(eval_undo) > 10:  # if there are too many undos
            # remove the oldest undo
            eval_undo.pop(0)
            pval_undo.pop(0)

        # add the current state to the undo memory
        eval_undo.append(suave.E)
        pval_undo.append(suave.P)

        # set the current state to the next state and remove the next state from the redo memory
        suave.E = copy.deepcopy(eval_redo.pop(-1))
        suave.P = copy.deepcopy(pval_redo.pop(-1))

        return True

    def store_update():
        """
        This function stores the current state of the plot
        :return None
        """
        nonlocal eval_undo, pval_undo, eval_redo, pval_redo
        if len(eval_undo) > 10:
            eval_undo.pop(0)
            pval_undo.pop(0)
        eval_undo.append(copy.deepcopy(suave.E))
        pval_undo.append(copy.deepcopy(suave.P))
        eval_redo = []
        pval_redo = []

    last_forward_sweep_click = 0  # initialize last forward sweep click
    last_backward_sweep_click = 0  # initialize last backward sweep click
    last_alternate_sweep_click = 0  # initialize last alternate sweep click
    last_shuffle_click = 0  # initialize last shuffle click
    last_redraw_click = 0  # initialize last redraw click
    last_swap_click = 0  # initialize last swap click
    last_stop_click = 0  # initialize last stop click
    last_undo_click = 0  # initialize last undo click
    last_redo_click = 0  # initialize last redo click
    last_remove_click = 0  # initialize last remove click
    last_reset_click = 0  # initialize last reset click
    callback_running = False  # initialize callback running check

    @app.callback(
        [Output('graph', 'figure'), Output('loading-reorder-out', 'children')],
        [Input('button-forwards', 'n_clicks'), Input('button-backwards', 'n_clicks'), Input('button-alternate', 'n_clicks'),
         Input('point-slider', 'value'), Input('state-slider', 'value'), Input('print-var', 'value'), Input('stencil-width', 'value'),
         Input('order-value', 'value'), Input('shuffle-button', 'n_clicks'),
         Input('interpolate', 'value'), Input('numSweeps', 'value'), Input('redraw', 'n_clicks'),
         Input('prop-list', 'value'),
         Input('max-pan', 'value'), Input('energy-width', 'value'), Input('redundant', 'value'),
         Input('energy-slider', 'value'),
         Input('swap-button', 'n_clicks'), Input('swap-input1', 'value'), Input('swap-input2', 'value'),
         Input('undo', 'n_clicks'),
         Input('stop-button', 'n_clicks'), Input('redo', 'n_clicks'), Input('remove-button', 'n_clicks'),
         Input('remove-state-input', 'value'),
         Input('reset-button', 'n_clicks')])
    def update_graph(forward_sweep_clicks,backward_sweep_clicks,alternate_sweep_clicks, point_bounds, state_bounds,
                     print_var, stencil_width, order, shuffle_clicks, interpolative, numSweeps, redraw_clicks,
                     prop_list, maxPan, energyWidth, redundant, energy_bounds, swap_clicks, swap1, swap2, undo_clicks,
                     stop_clicks, redo_clicks, remove_clicks, remove_state,
                     reset_clicks):
        """
        This function updates the graph

        :param forward_sweep_clicks:  the number of times the forward sweep button has been clicked
        :param backward_sweep_clicks:  the number of times the backwards sweep button has been clicked
        :param alternate_sweep_clicks:  the number of times the alternate sweep button has been clicked
        :param point_bounds:  the bounds of the points to be plotted
        :param state_bounds:  the bounds of the states to be plotted
        :param print_var:  the variable to be plotted
        :param stencil_width:  the width of the stencil to use
        :param order:  the order to use for each sweep
        :param shuffle_clicks:  the number of times the shuffle button has been clicked
        :param interpolative:  whether to interpolate
        :param numSweeps:  the number of sweeps to do
        :param redraw_clicks:  the number of times the redraw button has been clicked
        :param prop_list: the properties to enforce continuity for
        :param maxPan:  the maximum number of points to pan in stencil
        :param energyWidth:  the width of the energy window
        :param redundant:  whether to use redundant swaps
        :param energy_bounds:  the bounds of the energies to be plotted
        :param swap_clicks:  the number of times the swap button has been clicked
        :param swap1:  the first state to swap
        :param swap2:  the second state to swap
        :param undo_clicks:  the number of times the undo button has been clicked
        :param stop_clicks:  the number of times the stop button has been clicked
        :param redo_clicks:  the number of times the redo button has been clicked
        :param remove_clicks: the number of times the remove button has been clicked
        :param remove_state: the state to remove
        :param reset_clicks: the number of times the reset button has been clicked
        :return the figure to be plotted
        """
        nonlocal last_forward_sweep_click, last_backward_sweep_click, last_alternate_sweep_click, last_shuffle_click, last_redraw_click, last_swap_click
        nonlocal last_undo_click, last_stop_click, last_redo_click, last_remove_click, last_reset_click
        nonlocal eval_undo, pval_undo, eval_redo, pval_redo, original_eval, original_pval
        nonlocal sweep, callback_running

        if stop_clicks > last_stop_click:  # if stop button has been clicked
            last_stop_click = stop_clicks  # update last stop click
            suave.halt = True  # halt the scanner
            return make_figure()[0], "Aborted"
        else:
            suave.halt = False  # otherwise, don't halt the scanner

        if callback_running:  # if callback is running
            # update all click counters
            last_forward_sweep_click = forward_sweep_clicks
            last_backward_sweep_click = backward_sweep_clicks
            last_alternate_sweep_click = alternate_sweep_clicks
            last_shuffle_click = shuffle_clicks
            last_redraw_click = redraw_clicks
            last_swap_click = swap_clicks
            last_stop_click = stop_clicks
            last_undo_click = undo_clicks
            last_redo_click = redo_clicks
            last_remove_click = remove_clicks
            last_reset_click = reset_clicks
            raise PreventUpdate  # prevent update

        callback_running = True  # set callback running to true

        # assign values to global variables
        suave.pntBounds = point_bounds
        suave.stateBounds = state_bounds
        suave.propList = [int(prop) for prop in prop_list]
        suave.ignoreProps = len(suave.propList) == 0  # if no properties are selected, ignore properties
        suave.energyBounds = [float(energy) for energy in energy_bounds]
        suave.printVar = int(print_var)
        suave.width = int(stencil_width)
        suave.orders = [int(order)]  # only use one order for now
        suave.interpolate = bool(interpolative)
        suave.redundantSwaps = bool(redundant)
        suave.maxPan = int(maxPan)
        suave.energyWidth = float(energyWidth)

        # check input values
        if suave.printVar > suave.numProps:  # if print variable is greater than number of properties
            suave.printVar = 0  # set print variable to 0 (energy)
            print("Invalid print variable. Printing energy instead.", flush=True)

        if suave.orders[0] >= suave.numPoints:  # if order is greater than number of points
            suave.orders[0] = suave.numPoints - 1  # set order to number of points - 1
            print("Order too large. Using minimum order instead.", flush=True)

        if suave.width <= order:  # if stencil width is less than or equal to order
            suave.width = order + 1  # set stencil width to order + 1
            print("Stencil width too small. Using minimum stencil width instead.", flush=True)

        if suave.width >= suave.numPoints:  # if stencil width is greater than or equal to number of points
            suave.width = suave.numPoints - 1  # set stencil width to number of points - 1
            print("Stencil width too large. Using minimum stencil width instead.", flush=True)

        if suave.pntBounds[0] >= suave.pntBounds[1] - 1:  # if point bounds are invalid
            suave.pntBounds[1] = suave.pntBounds[0] + 1  # set point bounds to minimum
            print("Point bounds too small. Using minimum point bounds instead.", flush=True)
        if suave.pntBounds[1] <= suave.pntBounds[0] + 1:  # if point bounds are invalid
            suave.pntBounds[0] = suave.pntBounds[1] - 1  # set point bounds to minimum
            print("Point bounds too small. Using minimum point bounds instead.", flush=True)
        if suave.stateBounds[0] >= suave.stateBounds[1] - 1:  # if state bounds are invalid
            suave.stateBounds[1] = suave.stateBounds[0] + 1  # set state bounds to minimum
            print("State bounds too small. Using minimum state bounds instead.", flush=True)
        if suave.stateBounds[1] <= suave.stateBounds[0] + 1:  # if state bounds are invalid
            suave.stateBounds[0] = suave.stateBounds[1] - 1  # set state bounds to minimum
        if abs(suave.energyBounds[1] - suave.energyBounds[0]) <= 1e-6:  # if energy bounds are invalid
            min_bound = min(suave.energyBounds[0], suave.energyBounds[1])  # get minimum energy bound
            max_bound = max(suave.energyBounds[0], suave.energyBounds[1])  # get maximum energy bound
            suave.energyBounds[0] = min_bound - 1e-6  # set minimum energy bound
            suave.energyBounds[1] = max_bound + 1e-6  # set maximum energy bound
            print("Energy bounds too small. Using minimum energy bounds instead.", flush=True)

        back_sweep = False  # set back sweep to false
        alternate_sweep = False  # set alternate sweep to false

        # check which button was clicked and update the graph accordingly
        num_sweep_clicks = forward_sweep_clicks + backward_sweep_clicks + alternate_sweep_clicks
        last_num_sweep_clicks = last_forward_sweep_click + last_backward_sweep_click + last_alternate_sweep_click
        if num_sweep_clicks > last_num_sweep_clicks and num_sweep_clicks > 0:  # if sweep button was clicked
            # perform a sweep
            if forward_sweep_clicks > last_forward_sweep_click:
                back_sweep = False
            elif backward_sweep_clicks > last_backward_sweep_click:
                back_sweep = True
            elif alternate_sweep_clicks > last_alternate_sweep_click:
                alternate_sweep = True
            last_forward_sweep_click = forward_sweep_clicks
            last_backward_sweep_click = backward_sweep_clicks
            last_alternate_sweep_click = alternate_sweep_clicks

        elif redraw_clicks > last_redraw_click:  # redraw button clicked
            last_redraw_click = redraw_clicks
            ret = make_figure()[0], "Redrawn"
            callback_running = False
            return ret
        elif undo_clicks > last_undo_click:  # undo button clicked
            last_undo_click = undo_clicks

            if undo_callback():  # undo the last action
                ret = make_figure()[0], "Undone"  # update figure
            else:
                ret = no_update, "Nothing to undo"
            callback_running = False
            return ret
        elif redo_clicks > last_redo_click:
            last_redo_click = redo_clicks  # redo button clicked
            if redo_callback():  # redo the last action
                ret = make_figure()[0], "Redone"  # update figure
            else:
                ret = no_update, "Nothing to redo"
            callback_running = False
            return ret
        elif reset_clicks > last_reset_click:  # reset button clicked
            last_reset_click = reset_clicks
            suave.E = copy.deepcopy(original_eval)
            suave.P = copy.deepcopy(original_pval)
            ret = make_figure()[0], "Reset to Initial Data"  # update figure
            callback_running = False  # set callback running to false
            return ret  # return figure
        elif shuffle_clicks > last_shuffle_click:  # shuffle button clicked
            last_shuffle_click = shuffle_clicks  # update last shuffle click

            store_update()  # store current state
            suave.shuffle_energy()  # shuffle energy

            ret = make_figure()[0], "Shuffled"  # update figure
            callback_running = False  # set callback running to false
            return ret  # return figure
        elif swap_clicks > last_swap_click:  # swap button clicked
            last_swap_click = swap_clicks  # update last swap click

            store_update()  # store current state
            suave.E[[swap1, swap2], point_bounds[0]:point_bounds[1]] = suave.E[[swap2, swap1],
                                                                      point_bounds[0]:point_bounds[1]]
            suave.P[[swap1, swap2], point_bounds[0]:point_bounds[1]] = suave.P[[swap2, swap1],
                                                                      point_bounds[0]:point_bounds[1]]

            ret = make_figure()[0], "Swapped States {} and {} at Points {} to {}".format(swap1, swap2, suave.allPnts[
                point_bounds[0]], suave.allPnts[point_bounds[1] - 1])
            callback_running = False
            return ret
        elif remove_clicks > last_remove_click:  # remove button clicked
            last_remove_click = remove_clicks  # update last remove click
            remove_state = int(remove_state)  # get state to remove
            suave.E[remove_state, point_bounds[0]:point_bounds[1]] = np.nan  # set energy to nan
            suave.P[remove_state, point_bounds[0]:point_bounds[1], prop_list] = np.nan  # set properties to nan
            suave.hasMissing = True  # set has missing to true
            ret = make_figure()[0], "Removed State {} at Points {} to {} for selected properties".format(
                remove_state, suave.allPnts[point_bounds[0]], suave.allPnts[point_bounds[1] - 1])
            callback_running = False
            return ret
        else:  # otherwise, do nothing
            callback_running = False
            return no_update, no_update

        store_update()  # store current state

        # perform a sweep
        lastEvals, lastPvals = suave.prepareSweep()

        for i in range(numSweeps + numSweeps*int(alternate_sweep)): # perform sweeps and alternate sweeps if necessary
            lastEvals, lastPvals = suave.prepareSweep()
            sweep += 1
            for state in range(suave.numStates):
                suave.sweepState(state, sweep, backward=back_sweep)
            suave.analyzeSweep(lastEvals, lastPvals)
            if alternate_sweep:
                back_sweep = True
        time.sleep(0.1)

        delMax = suave.analyzeSweep(lastEvals, lastPvals)
        print("CONVERGENCE PROGRESS: {:e}".format(delMax), flush=True)

        if delMax < suave.tol:
            print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax), flush=True)
            suave.sortEnergies()

        # update plot data
        last_forward_sweep_click = forward_sweep_clicks
        last_backward_sweep_click = backward_sweep_clicks
        last_alternate_sweep_click = alternate_sweep_clicks
        ret = make_figure()
        callback_running = False
        return ret

    last_save_clicks = 0

    @app.callback(
        Output('loading-save-out', 'children'),
        [Input('save-button', 'n_clicks')])
    def save_order(save_clicks):
        """
        This function saves the order
        :param save_clicks:  the number of times the button has been clicked
        :return the bounds of the points to be plotted
        """
        nonlocal last_save_clicks, callback_running
        if callback_running:
            last_save_clicks = save_clicks
            raise PreventUpdate
        callback_running = True

        if save_clicks > last_save_clicks:
            lastEvals = copy.deepcopy(suave.E)
            lastPvals = copy.deepcopy(suave.P)

            if suave.hasMissing and suave.interpolate:
                if suave.keepInterp:
                    suave.interpMissing(interpKind="cubic")
            suave.saveOrder(isFinalResults=True)

            suave.E = copy.deepcopy(lastEvals)
            suave.P = copy.deepcopy(lastPvals)
            callback_running = False
            last_save_clicks = save_clicks
            return "Order saved"
        else:
            callback_running = False
            last_save_clicks = save_clicks
            return ""

    # run app without verbose output
    app.run_server(debug=False, use_reloader=False)