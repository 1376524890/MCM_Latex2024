from graphviz import Digraph

# ================= 字体设置 =================
NODE_FONT_SIZE = '26'  # 节点字体
EDGE_FONT_SIZE = '26'  # 边标签字体
OBSERVED_FONT_SIZE = '20'  # 观测变量字体

# ================= 颜色方案 =================
LATENT_COLOR = '#4B8BBE'         # 科技蓝
OBSERVED_COLOR = '#333333'       # 深灰边框
ERROR_COLOR = '#AAAAAA'          # 浅灰误差
SATISFACTION_COLOR = '#6C4B8E'   # 紫色核心变量

# ================= 观测变量定义（关键修复点） =================
observed_vars = {  # 必须在此处明确定义字典
    'Tourist_Q1': 'Tourist Q1',
    'Glacier_Q1': 'Glacier Q1',
    'Carbon_Q2': 'Carbon Q2'
}

# ================= 初始化图对象 =================
sem_diagram = Digraph(
    comment='Enhanced SEM Model',
    graph_attr={
        'rankdir': 'LR',
        'splines': 'ortho',
        'fontname': 'Helvetica',
        'nodesep': '1.0',
        'ranksep': '1.5',
        'compound': 'true'
    },
    node_attr={
        'fontsize': NODE_FONT_SIZE,
        'fontname': 'Helvetica',
        'margin': '0.4,0.2'
    },
    edge_attr={
        'fontsize': EDGE_FONT_SIZE,
        'fontname': 'Helvetica',
        'arrowsize': '1.2'
    }
)

# ================= 定义节点 =================
# 潜变量
latent_nodes = {
    'Carbon': {'fillcolor': LATENT_COLOR, 'fontcolor': 'white'},
    'Glacier': {'fillcolor': LATENT_COLOR, 'fontcolor': 'white'},
    'Tourist': {'fillcolor': LATENT_COLOR, 'fontcolor': 'white'},
    'Satisfaction': {'fillcolor': SATISFACTION_COLOR, 'fontcolor': 'white'}
}

for name, attrs in latent_nodes.items():
    sem_diagram.node(
        name, shape='circle', width='1.0', height='1.0',
        style='filled', color='black', label=name, **attrs
    )

# 观测变量（使用预定义的observed_vars）
for var, label in observed_vars.items():
    sem_diagram.node(
        var, label=label, shape='box', style='filled',
        fillcolor='white', color=OBSERVED_COLOR,
        width='1.6', height='0.8', fontsize=OBSERVED_FONT_SIZE,
        penwidth='1.2'
    )

# 误差项
errors = ['eTourist_Q1', 'eGlacier_Q1', 'eCarbon_Q2']
for e in errors:
    sem_diagram.node(
        e, shape='circle', width='0.5', height='0.5',
        style='filled', fillcolor=ERROR_COLOR, color=ERROR_COLOR
    )

# ================= 定义边 =================
# 测量模型
measurement_edges = [
    ('Tourist', 'Tourist_Q1', '0.718**'),
    ('Glacier', 'Glacier_Q1', '1.000'),
    ('Carbon', 'Carbon_Q2', '0.558*')
]

for src, tgt, coef in measurement_edges:
    sem_diagram.edge(
        src, tgt, label=coef, color=OBSERVED_COLOR,
        penwidth='1.5', arrowhead='normal'
    )

# 结构模型
structural_edges = [
    ('Carbon', 'Satisfaction', '-0.204'),
    ('Tourist', 'Satisfaction', '0.524*'),
    ('Glacier', 'Satisfaction', '0.269'),
    ('Satisfaction', 'Glacier_Q1', '0.587**'),
    ('Satisfaction', 'Carbon_Q2', '0.100')
]

for src, tgt, coef in structural_edges:
    sem_diagram.edge(
        src, tgt, label=coef,
        color=SATISFACTION_COLOR if src == 'Satisfaction' else LATENT_COLOR,
        penwidth='2.0', arrowhead='vee' if src == 'Satisfaction' else 'normal'
    )

# 误差项连接
error_edges = [
    ('eTourist_Q1', 'Tourist_Q1', '0.610'),
    ('eGlacier_Q1', 'Glacier_Q1', '0.000'),
    ('eCarbon_Q2', 'Carbon_Q2', '1.092')
]

for err, var, coef in error_edges:
    sem_diagram.edge(
        err, var, label=coef, style='dashed',
        color=ERROR_COLOR, penwidth='1.2', arrowsize='0.6'
    )

# ================= 布局优化 =================
with sem_diagram.subgraph() as s:
    s.attr(rank='same')
    s.node('Carbon')
    s.node('Glacier')
    s.node('Tourist')

for var, e in zip(observed_vars.keys(), errors):
    sem_diagram.edge(e, var, style='invis')

# ================= 输出图像 =================
sem_diagram.format = 'png'
sem_diagram.render('sem_diagram_final', view=True, cleanup=True)