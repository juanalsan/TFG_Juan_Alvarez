import pandas as pd
import numpy as np
import random
import math
import time


def Genera_solucion_vecina(solucion_actual, rangos):
    salto=[400,2]
    sol = list(solucion_actual)
    sol = [sol[i] + random.randint(-salto[i], salto[i]) for i in range(len(sol))]
    sol = [np.clip(sol[i], rangos[i][0], rangos[i][1]) for i in range(len(sol))]
    return sol

def evaluar(sol, df):
    return -df.iloc[sol[0], sol[1]]

def temple_simulado_con_recalentamiento(df, alpha=0.95, c_0=60000, rep_max=150, it_base=200, n_rec=5):
    start_time = time.time()  
    c, rep = c_0, 0
    criterio_parada = True
    
    rangos = [[0, df.shape[0] - 1], [0, df.shape[1] - 1]]
    
    
    s_mejor = s_actual = [random.randint(rangos[i][0], rangos[i][1]) for i in range(2)]
    f_mejor = f_actual = evaluar(s_actual, df)
    while criterio_parada:
        iteraciones = it_base * 2 if c > 20000 else it_base
        for _ in range(iteraciones):
            if rep >= rep_max or c < 0.1:
                if n_rec > 0:
                    n_rec -= 1
                    c += 250 * n_rec ** 2
                    s_actual, f_actual, rep = s_mejor, f_mejor, 0
                else:
                    criterio_parada = False
                break
            
            s_nueva = Genera_solucion_vecina(s_actual, rangos)
            f_nueva = evaluar(s_nueva, df)
            dif = f_actual - f_nueva
            
            prob = math.exp(np.clip(dif / c, -100, 100))
            
            if dif > 0 or random.random() < prob:
                s_actual, f_actual = s_nueva, f_nueva
                rep = 0
                if f_mejor - f_nueva > 0:
                    s_mejor, f_mejor = s_nueva, f_nueva
            else:
                rep += 1
        
        c *= alpha

    end_time = time.time()  
    execution_time = end_time - start_time  
    
    return s_mejor, -f_mejor, execution_time