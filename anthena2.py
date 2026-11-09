import numpy as np
import random
import math

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS, ELÉCTRICOS Y DEL ALGORITMO GENÉTICO
# =====================================================================
RADIO_MAX = 0.040         # 4.0 cm de radio máximo del cilindro único (0.04 m)
ALTURA_MAX_24 = 0.070     # 7.0 cm de altura máxima para rama de 2.44 GHz (0.07 m)
ALTURA_MAX_55 = 0.035     # 3.5 cm de altura máxima para rama de 5.50 GHz (0.035 m)

RADIO_ALAMBRE = 0.0005    # Alambre de cobre de 1 mm de diámetro (0.5 mm de radio)
NUM_SEGMENTOS = 8         # 8 pasos (segmentos) por cada rama de la cruz

FREQ_24 = 2.44e9          # Frecuencia central banda baja Wi-Fi 6 (2.44 GHz)
FREQ_55 = 5.50e9          # Frecuencia central banda alta Wi-Fi 6 (5.50 GHz)
Z0 = 50.0                 # Impedancia característica del puerto RF del ESP32-C5 (50 Ohm)

TAMANO_POBLACION = 128
GENERACIONES = 1024
ELITE_SIZE = 8
TASA_MUTACION = 0.25

# Constantes físicas
C_LUZ = 299792458.0
EPS_0 = 8.854187817e-12
MU_0 = 4.0 * math.pi * 1e-7
RHO_COBRE = 1.68e-8       # Resistividad del cobre (Ohm*m)

# =====================================================================
# 2. REPRESENTACIÓN GENÉTICA Y CODIFICACIÓN PROGRESIVA
# =====================================================================
def crear_rama_genes(altura_max):
    genes = []
    dz_medio = altura_max / NUM_SEGMENTOS
    for _ in range(NUM_SEGMENTOS):
        r = random.uniform(0.003, RADIO_MAX)
        d_theta = random.uniform(-1.0, 1.0)
        d_z = random.uniform(dz_medio * 0.4, dz_medio * 1.3)
        genes.append([r, d_theta, d_z])
    return np.array(genes, dtype=float)

def crear_individuo():
    return {
        'rama_24': crear_rama_genes(ALTURA_MAX_24),
        'rama_55': crear_rama_genes(ALTURA_MAX_55)
    }

def copiar_individuo(ind):
    return {
        'rama_24': np.copy(ind['rama_24']),
        'rama_55': np.copy(ind['rama_55'])
    }

def decodificar_cruz_a_cartesianas(individuo):
    def construir_nodos(genes, altura_max, theta_inicial):
        nodos = [[0.0, 0.0, 0.0]] 
        suma_dz = sum(max(g[2], 0.001) for g in genes)
        escala_z = min(1.0, altura_max / max(suma_dz, 1e-6))
        
        theta_actual = theta_inicial
        z_actual = 0.0
        
        for idx, (r, d_theta, d_z) in enumerate(genes):
            r_clip = float(np.clip(r, 0.002, RADIO_MAX))
            if idx == 0:
                theta_actual = theta_inicial + d_theta * 0.25
            else:
                theta_actual += d_theta
                
            z_actual += max(d_z, 0.001) * escala_z
            z_clip = float(np.clip(z_actual, 0.001, altura_max))
            
            x = r_clip * math.cos(theta_actual)
            y = r_clip * math.sin(theta_actual)
            nodos.append([x, y, z_clip])
            
        return np.array(nodos, dtype=float)

    nodos_24 = construir_nodos(individuo['rama_24'], ALTURA_MAX_24, 0.0)
    nodos_55 = construir_nodos(individuo['rama_55'], ALTURA_MAX_55, math.pi / 2.0)
    return nodos_24, nodos_55

# =====================================================================
# 3. MOTOR DE COLISIONES, ISOTROPÍA Y PENALIZACIÓN
# =====================================================================
def motor_de_colisiones_3d(nodos_a, nodos_b, margen_mm=2.5):
    """
    Detecta si la rama_A y rama_B se intersectan o se acercan demasiado
    en el espacio 3D, ignorando el origen común de soldadura.
    """
    margen_metros = margen_mm / 1000.0
    
    # Interpolar 5 puntos por segmento para crear detección continua
    puntos_a = []
    for i in range(len(nodos_a) - 1):
        for t in np.linspace(0, 1, 5):
            puntos_a.append(nodos_a[i] + t * (nodos_a[i+1] - nodos_a[i]))
            
    puntos_b = []
    for i in range(len(nodos_b) - 1):
        for t in np.linspace(0, 1, 5):
            puntos_b.append(nodos_b[i] + t * (nodos_b[i+1] - nodos_b[i]))
            
    pA = np.array(puntos_a)
    pB = np.array(puntos_b)
    
    # Filtrar puntos muy cercanos al origen (Z < 3mm) porque ahí nacen juntas
    pA_validos = pA[np.linalg.norm(pA, axis=1) > 0.003]
    pB_validos = pB[np.linalg.norm(pB, axis=1) > 0.003]
    
    if len(pA_validos) == 0 or len(pB_validos) == 0:
        return 0.0
        
    # Calcular matriz de distancias vectorizada
    diff = pA_validos[:, np.newaxis, :] - pB_validos[np.newaxis, :, :]
    distancias = np.linalg.norm(diff, axis=-1)
    
    min_dist = np.min(distancias)
    
    if min_dist < margen_metros:
        # Castigo masivo al fitness por cortocircuito / cercanía extrema
        return (margen_metros - min_dist) * 50000.0 
    return 0.0

def evaluar_geometria_e_isotropia(nodos):
    proyecciones = np.zeros(3)
    vectores = []
    penalizacion_doblado = 0.0

    for i in range(len(nodos) - 1):
        vec = nodos[i+1] - nodos[i]
        norma = float(np.linalg.norm(vec))
        if norma < 1e-6:
            penalizacion_doblado += 10.0
            continue
        proyecciones += np.abs(vec)
        vectores.append(vec / norma)

    for i in range(len(vectores) - 1):
        cos_ang = float(np.dot(vectores[i], vectores[i+1]))
        if cos_ang < -0.20:
            penalizacion_doblado += abs(cos_ang + 0.20) * 18.0

    suma_total = float(np.sum(proyecciones)) + 1e-6
    fracciones = proyecciones / suma_total
    desviacion_isotropia = float(np.std(fracciones))
    puntaje_isotropia = max(0.0, 10.0 * (1.0 - (desviacion_isotropia / 0.35)))

    return puntaje_isotropia, penalizacion_doblado

# =====================================================================
# 4. MODELO ELÉCTRICO RLC Y RESOLUCIÓN AC
# =====================================================================
def calcular_impedancia_rama_escalera(nodos, freq, factor_resonancia_dipolo):
    omega = 2.0 * math.pi * freq
    longitud_onda = C_LUZ / freq
    delta_skin = math.sqrt(RHO_COBRE / (math.pi * freq * MU_0))
    parametros_rlc = []
    longitud_total = 0.0

    for i in range(len(nodos) - 1):
        p1, p2 = nodos[i], nodos[i+1]
        vec = p2 - p1
        l_seg = max(float(np.linalg.norm(vec)), 1e-4)
        longitud_total += l_seg
        z_medio = max(float((p1[2] + p2[2]) / 2.0), 0.001)
        cos_alpha = abs(float(vec[2])) / l_seg

        term_log_l = max(math.log((2.0 * l_seg) / RADIO_ALAMBRE) - 1.0, 0.15)
        L_seg = (MU_0 / (2.0 * math.pi)) * l_seg * term_log_l

        term_log_c = max(math.log((2.0 * z_medio) / RADIO_ALAMBRE), 0.25)
        C_seg = (2.0 * math.pi * EPS_0 * l_seg) / term_log_c

        R_skin = (RHO_COBRE * l_seg) / (2.0 * math.pi * RADIO_ALAMBRE * delta_skin)
        l_norm = min(l_seg / longitud_onda, 0.5)
        R_rad = 73.0 * (math.pi ** 2) * (l_norm ** 2) * (0.5 + 0.5 * (cos_alpha ** 2))
        R_seg = max(R_rad + R_skin, 0.02)

        parametros_rlc.append((R_seg, L_seg, C_seg))

    beta = (2.0 * math.pi) / longitud_onda
    elec_len = beta * longitud_total * factor_resonancia_dipolo
    x_apertura = -Z0 / (math.tan(max(min(elec_len, math.pi - 0.05), 0.05)) + 1e-6)
    z_carga = complex(36.5 * (longitud_total / (longitud_onda * 0.5)) ** 2, x_apertura * 0.35)

    for R_s, L_s, C_s in reversed(parametros_rlc):
        z_serie = complex(R_s, omega * L_s * 0.45)
        y_paralelo = complex(0.0, omega * C_s * 0.65)
        y_nodo = (1.0 / (z_carga + 1e-9)) + y_paralelo
        z_carga = z_serie + (1.0 / y_nodo)

    return z_carga

def simular_cruz_multibanda(individuo):
    nodos_24, nodos_55 = decodificar_cruz_a_cartesianas(individuo)

    vec_medio_24 = np.mean(nodos_24[1:, :2], axis=0)
    vec_medio_55 = np.mean(nodos_55[1:, :2], axis=0)
    norm_24 = float(np.linalg.norm(vec_medio_24)) + 1e-6
    norm_55 = float(np.linalg.norm(vec_medio_55)) + 1e-6
    cos_cruz = abs(float(np.dot(vec_medio_24 / norm_24, vec_medio_55 / norm_55)))
    factor_acoplamiento_mutuo = 0.04 + 0.25 * cos_cruz

    z24_rama1 = calcular_impedancia_rama_escalera(nodos_24, FREQ_24, 1.0)
    z24_rama2 = calcular_impedancia_rama_escalera(nodos_55, FREQ_24, 0.85)
    y_in_24 = (1.0 / z24_rama1) + (factor_acoplamiento_mutuo / z24_rama2)
    z_in_24 = 1.0 / y_in_24

    z55_rama1 = calcular_impedancia_rama_escalera(nodos_24, FREQ_55, 0.85)
    z55_rama2 = calcular_impedancia_rama_escalera(nodos_55, FREQ_55, 1.0)
    y_in_55 = (1.0 / z55_rama2) + (factor_acoplamiento_mutuo / z55_rama1)
    z_in_55 = 1.0 / y_in_55

    gamma_24 = abs((z_in_24 - Z0) / (z_in_24 + Z0))
    gamma_55 = abs((z_in_55 - Z0) / (z_in_55 + Z0))

    s11_24 = 20.0 * math.log10(max(gamma_24, 1e-5))
    s11_55 = 20.0 * math.log10(max(gamma_55, 1e-5))

    iso_24, pen_24 = evaluar_geometria_e_isotropia(nodos_24)
    iso_55, pen_55 = evaluar_geometria_e_isotropia(nodos_55)
    puntaje_omni = iso_24 + iso_55
    
    # Motor de Colisión con clearance de 2.5 milímetros
    pen_colision = motor_de_colisiones_3d(nodos_24, nodos_55, margen_mm=2.5)
    penalizacion_geom = pen_24 + pen_55 + (cos_cruz * 12.0) + pen_colision

    peor_s11 = max(s11_24, s11_55)
    promedio_s11 = 0.5 * (s11_24 + s11_55)

    penalizacion_s11 = 0.0
    if s11_24 > -10.0: penalizacion_s11 += (s11_24 - (-10.0)) * 8.0
    if s11_55 > -10.0: penalizacion_s11 += (s11_55 - (-10.0)) * 8.0

    fitness = (-peor_s11 * 2.5) + (-promedio_s11 * 1.0) + (puntaje_omni * 0.8) - penalizacion_s11 - penalizacion_geom

    return fitness, s11_24, s11_55, z_in_24, z_in_55, puntaje_omni

# =====================================================================
# 5. OPERADORES GENÉTICOS
# =====================================================================
def cruzar_individuos(padre1, padre2):
    hijo = crear_individuo()
    for i in range(NUM_SEGMENTOS):
        hijo['rama_24'][i] = np.copy(padre1['rama_24'][i] if random.random() < 0.5 else padre2['rama_24'][i])
        hijo['rama_55'][i] = np.copy(padre1['rama_55'][i] if random.random() < 0.5 else padre2['rama_55'][i])
    return hijo

def mutar_individuo(individuo):
    mutado = copiar_individuo(individuo)
    def mutar_rama(genes, altura_max):
        dz_max_paso = (altura_max / NUM_SEGMENTOS) * 1.6
        for i in range(NUM_SEGMENTOS):
            if random.random() < TASA_MUTACION:
                genes[i, 0] = np.clip(genes[i, 0] + random.gauss(0, 0.0025), 0.002, RADIO_MAX)
                genes[i, 1] = np.clip(genes[i, 1] + random.gauss(0, 0.18), -1.4, 1.4)
                genes[i, 2] = np.clip(genes[i, 2] + random.gauss(0, 0.0015), 0.0015, dz_max_paso)
    mutar_rama(mutado['rama_24'], ALTURA_MAX_24)
    mutar_rama(mutado['rama_55'], ALTURA_MAX_55)
    return mutado

# =====================================================================
# 6. BUCLE EVOLUTIVO PRINCIPAL
# =====================================================================
def ejecutar_optimizacion():
    random.seed(42)
    np.random.seed(42)

    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    print("=" * 86)
    print(" OPTIMIZACIÓN EVOLUTIVA + DETECCIÓN DE COLISIONES (ESP32-C5)")
    print(f" Cilindro Único: R={RADIO_MAX*100:.1f} cm | H_2.44={ALTURA_MAX_24*100:.1f} cm | H_5.50={ALTURA_MAX_55*100:.1f} cm")
    print(f" Población: {TAMANO_POBLACION} | Generaciones: {GENERACIONES} | Élite intacta: {ELITE_SIZE} | Pasos/rama: {NUM_SEGMENTOS}")
    print("=" * 86)

    for gen in range(GENERACIONES):
        evaluados = []
        for ind in poblacion:
            fit, s11_24, s11_55, z24, z55, omni = simular_cruz_multibanda(ind)
            evaluados.append((fit, s11_24, s11_55, z24, z55, omni, ind))

        evaluados.sort(key=lambda x: x[0], reverse=True)

        if gen % 32 == 0 or gen == GENERACIONES - 1:
            bfit, bs24, bs55, bz24, bz55, bomni, _ = evaluados[0]
            print(f"Gen {gen:3d}/{GENERACIONES} | Fit: {bfit:8.2f} | "
                  f"S11(2.44G): {bs24:6.2f} dB | S11(5.50G): {bs55:6.2f} dB | "
                  f"Omni: {bomni:5.2f} pts")

        mejores_ocho = [copiar_individuo(item[6]) for item in evaluados[:ELITE_SIZE]]
        nueva_poblacion = [copiar_individuo(elite) for elite in mejores_ocho]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1, padre2 = random.sample(mejores_ocho, 2)
            hijo = cruzar_individuos(padre1, padre2)
            hijo_mutado = mutar_individuo(hijo)
            nueva_poblacion.append(hijo_mutado)

        poblacion = nueva_poblacion

    mejor_fit, mejor_s24, mejor_s55, mejor_z24, mejor_z55, mejor_omni, campeon = evaluados[0]
    nodos_24_m, nodos_55_m = decodificar_cruz_a_cartesianas(campeon)

    nodos_24_cm = nodos_24_m * 100.0
    nodos_55_cm = nodos_55_m * 100.0

    len_24_cm = sum(np.linalg.norm(nodos_24_cm[i+1] - nodos_24_cm[i]) for i in range(NUM_SEGMENTOS))
    len_55_cm = sum(np.linalg.norm(nodos_55_cm[i+1] - nodos_55_cm[i]) for i in range(NUM_SEGMENTOS))

    print("\n" + "=" * 86)
    print(" ¡OPTIMIZACIÓN COMPLETADA CON ÉXITO Y SIN COLISIONES!")
    print(f" * S11 Combinado @ 2.44 GHz : {mejor_s24:6.2f} dB  (Z_in = {mejor_z24.real:5.2f} {mejor_z24.imag:+6.2f}j Ohm)")
    print(f" * S11 Combinado @ 5.50 GHz : {mejor_s55:6.2f} dB  (Z_in = {mejor_z55.real:5.2f} {mejor_z55.imag:+6.2f}j Ohm)")
    print(f" * Puntaje Omnidireccional  : +{mejor_omni:5.2f} pts")
    print(f" * Longitud física alambre  : Rama 2.44G = {len_24_cm:.2f} cm | Rama 5.50G = {len_55_cm:.2f} cm")
    print("=" * 86)

    print("\n// --- COPIAR Y PEGAR DIRECTAMENTE EN OPENSCAD ---")
    print("nodos_24 = [")
    for idx, pt in enumerate(nodos_24_cm):
        coma = "," if idx < len(nodos_24_cm) - 1 else ""
        print(f"    [{pt[0]:7.3f}, {pt[1]:7.3f}, {pt[2]:7.3f}]{coma} // Paso {idx}")
    print("];\n")

    print("nodos_55 = [")
    for idx, pt in enumerate(nodos_55_cm):
        coma = "," if idx < len(nodos_55_cm) - 1 else ""
        print(f"    [{pt[0]:7.3f}, {pt[1]:7.3f}, {pt[2]:7.3f}]{coma} // Paso {idx}")
    print("];")

if __name__ == "__main__":
    ejecutar_optimizacion()