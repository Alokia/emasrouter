from dash import Dash, dcc, html, Input, Output, ctx, callback
import dash_cytoscape as cyto
from typing import List, Dict
import webbrowser
import multiprocessing

__all__ = ["show_dash_graph"]

cyto.load_extra_layouts()

app = Dash(__name__)
server = app.server

stylesheet = [
    {
        'selector': 'edge',
        'style': {
            'label': 'data(label)',
            # 'color': 'black',
            # 'font-size': 18,
            'text-valign': 'top',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle-backcurve',
            'control-point-step-size': 100,  # 边弯曲程度
            'width': 3,
        }
    },
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'width': 60,
            'height': 60,
            'text-valign': 'center',
            'text-halign': 'center',
            # 'background-color': 'blue'
        }
    },
    {
        'selector': '.input',
        'style': {
            'background-color': '#0072B2'
        }
    },
    {
        'selector': '.output',
        'style': {
            'background-color': '#E69F00'
        }
    },
    {
        'selector': '.midpoint',
        'style': {
            'background-color': '#90C987',
        }
    },
    {
        'selector': '.single',
        'style': {
            'background-color': '#636e72',
        }
    }
]


def show_dash_graph(elements: List[Dict[str, Dict[str, str]]]):
    process = multiprocessing.Process(target=_show_graph, args=(elements,))
    process.start()
    webbrowser.open('http://127.0.0.1:8050')
    return process


def _show_graph(elements: List[Dict[str, Dict[str, str]]]):
    app.layout = html.Div([
        html.Div(className="download-container", children=[
            html.Div('Download graph:'),
            dcc.Tabs(id='tabs-image-export', children=[
                dcc.Tab(label='generate jpg', value='jpg'),
                dcc.Tab(label='generate png', value='png'),
                dcc.Tab(label='generate svg', value='svg'),
            ], style={'display': 'none'}),
            html.Button("as jpg", id="btn-get-jpg"),
            html.Button("as png", id="btn-get-png"),
            html.Button("as svg", id="btn-get-svg"),
        ], style={"textAlign": "center", 'margin': '0 auto'}),

        html.Div(children=[
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'loop-direction:',
                dcc.Input(id='input-loop-direction', type='text', value='30deg')
            ]),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'loop-sweep:',
                dcc.Input(id='input-loop-sweep', type='text', value='-90deg')
            ]),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'text-margin-y:',
                dcc.Input(id='input-text-margin-y', type='number', value=-18)
            ]),
            html.Br(),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'input-color:',
                dcc.Input(id='input-background-color', type='text', value='#0072B2')
            ]),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'output-color:',
                dcc.Input(id='output-background-color', type='text', value='#E69F00')
            ]),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'mid-color:',
                dcc.Input(id='mid-background-color', type='text', value='#90C987')
            ]),
            html.Div(style={'width': '50%', 'display': 'inline'}, children=[
                'single-color:',
                dcc.Input(id='single-background-color', type='text', value='#636e72')
            ]),
        ], style={"textAlign": "center", 'margin': '0 auto'}),

        html.Div(className="dropdown-container", children=[
            dcc.Dropdown(
                id='dropdown-update-layout',
                value='grid',
                clearable=False,
                options=[
                    {'label': name.capitalize(), 'value': name}
                    for name in ['grid', 'random', 'circle', 'cose', 'concentric']
                ]
            ),
        ], style={"textAlign": "center", 'margin': '0 auto'}),

        html.Div([
            cyto.Cytoscape(
                id='graph',
                elements=elements,
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '600px'},
                stylesheet=stylesheet
            ),
        ])
    ])

    app.run(debug=True, host="127.0.0.1", port="8050", use_reloader=False)


@callback(Output("graph", "generateImage"),
          [Input('tabs-image-export', 'value'),
           Input("btn-get-jpg", "n_clicks"),
           Input("btn-get-png", "n_clicks"),
           Input("btn-get-svg", "n_clicks"), ])
def get_image(tab, get_jpg_clicks, get_png_clicks, get_svg_clicks):
    ftype = tab

    action = 'store'

    if ctx.triggered:
        if ctx.triggered_id != "tabs-image-export":
            action = "download"
            ftype = ctx.triggered_id.split("-")[-1]

    return {
        'type': ftype,
        'action': action
    }


@callback(Output('graph', 'layout'),
          Input('dropdown-update-layout', 'value'))
def update_layout(layout):
    return {
        'name': layout,
        'animate': True
    }


@callback(Output('graph', 'stylesheet'),
          [Input('input-loop-direction', 'value'),
           Input('input-loop-sweep', 'value'),
           Input("input-text-margin-y", "value"),
           Input("input-background-color", "value"),
           Input("output-background-color", "value"),
           Input("mid-background-color", "value"),
           Input("single-background-color", "value")])
def update_style(loop_direction, loop_sweep, text_margin_y, input_color, output_color, mid_color, single_color):
    for style in stylesheet:
        if style['selector'] == 'edge':
            style['style']['loop-direction'] = loop_direction
            style['style']['loop-sweep'] = loop_sweep
            style['style']['text-margin-y'] = text_margin_y
        elif style['selector'] == '.input':
            style['style']['background-color'] = input_color
        elif style['selector'] == '.output':
            style['style']['background-color'] = output_color
        elif style['selector'] == '.midpoint':
            style['style']['background-color'] = mid_color
        elif style['selector'] == '.single':
            style['style']['background-color'] = single_color
    return stylesheet
