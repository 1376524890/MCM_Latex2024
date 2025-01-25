from semopy import ModelGraph

# 假设 model 是你的 SEM 模型
g = ModelGraph(model, filename='optimized_sem_model', format='png')
g.render(view=True)