import pandas as pd
import numpy as np
import random
import time


def evaluar(sol, df):
    return df.iloc[sol[0], sol[1]]

def Genera_solucion_vecina(solucion_actual, rangos):
    salto_vertical = 400

    sol = list(solucion_actual)
    sol[0] += random.randint(-salto_vertical, salto_vertical)
    sol[1] = random.randint(rangos[1][0],rangos[1][1])
    sol[0] = np.clip(sol[0], rangos[0][0], rangos[0][1])

    return sol

# Crea la estructura de memoria a largo plazo 
#contiene las iteraciones y la mejor solucion de cada sección
def inicializar_LTM(f_LTM, c_LTM):
    return [[['None', -np.inf, 0, 0] for _ in range(c_LTM)] 
    for _ in range(f_LTM)]



# A partir de una solución busca en que sección de LTM está
def seccion_LTM(sol, f_LTM, c_LTM, f_csv, c_csv):
    fila = min(sol[0] // (f_csv // f_LTM), f_LTM - 1)
    columna = min(sol[1] // (c_csv // c_LTM), c_LTM - 1)
    return fila, columna


# Seleccionar una sección no visitada o si todas han sido visitadas,
# seleccionar una aleatoria
def seleccion_aleatoria_no_visitada_LTM(LTM, f_LTM=3, c_LTM=3):
    opciones = [(i, j) for i in range(f_LTM) for j in range(c_LTM) 
    if LTM[i][j][2] == 0]
    
    if opciones:
        return random.choice(opciones)
    else:
        return (random.randint(0, f_LTM-1), random.randint(0, c_LTM-1))
        
# Añadir la función correcta seleccion_aleatoria_no_visitada_LTM con parámetros
def seleccion_aleatoria_no_visitada_LTM(LTM, f_LTM, c_LTM):
    opciones = [(i, j) for i in range(f_LTM) for j in range(c_LTM) if LTM[i][j][2] == 0]
    if opciones:
        return random.choice(opciones)
    else:
        return (random.randint(0, f_LTM-1), random.randint(0, c_LTM-1))
    
# Genera los extremos de las secciones delimitadas por LMT
def generar_rangos_secciones(f_LTM, c_LTM, f_csv, c_csv):
    rangos_sec = []

    salto_v = f_csv // f_LTM
    salto_h = c_csv // c_LTM

    for i in range(f_LTM):
        inicio_v = i * salto_v
        fin_v = inicio_v + salto_v - 1 if i != f_LTM - 1 else f_csv - 1 
        for j in range(c_LTM):
            inicio_h = j * salto_h
            fin_h = inicio_h + salto_h - 1 if j != c_LTM - 1 else c_csv - 1 
            rangos_sec.append([[inicio_v, fin_v], [inicio_h, fin_h]])

    return rangos_sec



# Actualiza LMT añadiendo iteraciones y 
#cambiando la mejor solución de una sección si se encontró una
def actualizar_LTM(LTM, f, c, s_actual, f_actual, bool_ten=False,
LTM_ten=2):
    
    if f_actual >= LTM[f][c][1]:
        LTM[f][c] = [s_actual, f_actual, LTM[f][c][2]+1, LTM[f][c][3]] 
    else:
        LTM[f][c][2] += 1
    
    if bool_ten:
        #genero una lista de todos los indices de LTM
        indices = [[i, j] 
                   for i in range(len(LTM)) for j in range(len(LTM[0]))]
        #Recorro los indices y reduzco en 1 a todas las tenencias 
        for loc in [ind for ind in indices 
        if LTM[ind[0]][ind[1]][3] > 0 ]:
            LTM[loc[0]][loc[1]][3] -= 1
            
        LTM[f][c][3] += LTM_ten
            
    return LTM

# Búsqueda en una sección con memoria a corto plazo
def busqueda_tabu_mini(s_actual, LTM, df, seccion, rangos, bool_ten,
STM_len=10, n_vecinos=5, max_iter_TM=3):
    f_actual = evaluar(s_actual, df)

    lista_tabu = [s_actual]
    fila = seccion // len(LTM)
    columna = seccion % len(LTM[0])
    LTM = actualizar_LTM(LTM, fila, columna, s_actual, f_actual, 
    bool_ten)
    
    for _ in range(max_iter_TM):
        vecinos = [Genera_solucion_vecina(s_actual, rangos) for _ in 
        range(n_vecinos)]
        vecinos = [(v, evaluar(v, df)) for v in vecinos if v not in
        lista_tabu]
        
        if vecinos:
            s_actual, f_actual = max(vecinos, key=lambda x: x[1])
            lista_tabu.append(s_actual)
            if len(lista_tabu) > STM_len:
                lista_tabu.pop(0)
            LTM = actualizar_LTM(LTM, fila, columna, s_actual, f_actual)
                
    return s_actual, LTM


def busqueda_tabu_con_LTM(df, f_LTM=3, c_LTM=3, max_iter=200,
max_iter_TM=50, LTM_ten=2, STM_len = 10, n_vecinos=5):
    f_csv, c_csv= df.shape[0] , df.shape[1] 
    rangos_secciones = generar_rangos_secciones(f_LTM, c_LTM, f_csv, c_csv)
    start_time = time.time()
    rangos = [[0, f_csv-1], [0, c_csv-1]]
    
    s_mejor = s_actual = [random.randint(rangos[0][0], rangos[0][1]),
    random.randint(rangos[1][0], rangos[1][1])]
    f_mejor = f_actual = evaluar(s_actual, df)

    fila, columna = seccion_LTM(s_actual, f_LTM, c_LTM, f_csv, c_csv)
    LTM = inicializar_LTM(f_LTM, c_LTM)
    LTM = actualizar_LTM(LTM, fila, columna, s_actual, f_actual)
    rangos = rangos_secciones[fila*c_LTM + columna]
    
    cont, iter_d = 1, 5

    fase_actual = "Intensificación"
    seccion = fila * c_LTM + columna  

    for iter_num in range(max_iter):

        if fase_actual == "Diversificación":
            if cont < f_LTM * c_LTM: 
                cont+=1
                fila, columna = seleccion_aleatoria_no_visitada_LTM(LTM,
                                                                    f_LTM, c_LTM)     
            else:
                iter_d=max_iter_TM
                sec_no_tabu = [d for fila in LTM for d in fila if 
                d[3] == 0]
                mejor_sol = max(sec_no_tabu, key=lambda x: x[1])
                fila, columna = seccion_LTM(mejor_sol[0], f_LTM, c_LTM, f_csv, c_csv)
                  
            seccion = fila * c_LTM + columna  
            rangos = rangos_secciones[seccion]
            s_actual = [random.randint(rangos[0][0], rangos[0][1]), 
            random.randint(rangos[1][0], rangos[1][1])]
            s_actual, LTM = busqueda_tabu_mini(s_actual, LTM, df, 
            seccion, rangos, False, STM_len, n_vecinos, iter_d)    
            if f_mejor < LTM[fila][columna][1]:
                [s_mejor, f_mejor, _, _] = LTM[fila][columna]
            
            fase_actual = "Intensificación"
            
            
            
        elif fase_actual == "Intensificación":
            if cont < f_LTM * c_LTM:
                s_actual, LTM = busqueda_tabu_mini(s_actual, 
                LTM, df, seccion, rangos, STM_len, n_vecinos)
                
            else: 
                sec_no_tabu = [d for fila in LTM for d in fila if 
                d[3] == 0]
                mejor_sol = max(sec_no_tabu, key=lambda x: x[1])
                [s_actual, f_actual, _, _] = mejor_sol
                fila, columna = seccion_LTM(s_actual, f_LTM, c_LTM, f_csv, c_csv)
                seccion = fila * c_LTM + columna  
                rangos = rangos_secciones[seccion]
                
                s_actual, LTM = busqueda_tabu_mini(s_actual, 
                LTM, df, seccion, rangos, True, STM_len, n_vecinos, max_iter_TM)
            
            if f_mejor < LTM[fila][columna][1]:
                [s_mejor, f_mejor, _, _] = LTM[fila][columna]
            
            fase_actual = "Diversificación"
                
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
   

    return s_mejor, f_mejor, tiempo_ejecucion
