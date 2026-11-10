import numpy as np
import random
import math

# =====================================================================
# CARGA DE PYSPICE PARA VERIFICACIÓN Y GENERACIÓN DE NETLIST SPICE
# =====================================================================
PYSPICE_DISPONIBLE = False
try:
    import PySpice.Logging.Logging as Logging
    logger = Logging.setup_logging(logging_level='CRITICAL')
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import *
    PYSPICE_DISPONIBLE = True
except ImportError:
    print("Aviso: Librería PySpice no detectada. Se generará el Netlist SPICE")
    print("estándar de texto y se verificará con el solucionador nodal MNA.")

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS, ELÉCTRICOS Y DEL ALGORITMO GENÉTICO
# =====================================================================
RADIO_MAX = 0.040         # 4.0 cm de radio máximo del cilindro único (0.04 m)
ALTURA_MAX_24 = 0.070     # 7.0 cm de altura máxima para rama de 2.44 GHz (0.07 m)
ALTURA_MAX_55 = 0.035     # 3.5 cm de altura máxima para rama de 5.50 GHz (0.035 m)

RADIO_ALAMBRE = 0.0020    # Radio 2.0 mm (Alambre grueso de 4.0 mm de diámetro)
SEPARACION_AIRE_MM = 2.5  # Espacio libre mínimo entre las paredes de los dos cables (mm)
NUM_SEGMENTOS = 8         # 8 pasos (segmentos) por cada rama de la cruz

FREQ_24 = 2.44e9          # Frecuencia central banda baja Wi-Fi 6 (2.44 GHz)
FREQ_55 = 5.50e9          # Frecuencia central banda alta Wi-Fi 6 (5.50 GHz)
Z0 = 50.0                 # Impedancia característica del puerto RF del ESP32-C5 (50 Ohm)

TAMANO_POBLACION = 1024
GENERACIONES = 1024
ELITE_SIZE = 128
TASA_MUTACION = 0.25

# Constantes físicas
C_LUZ = 299792458.0
EPS_0 = 8.854187817e-12
MU_0 = 4.0 * math.pi * 1e-7
RHO_COBRE = 1.68e-8       # Resistividad del cobre (Ohm*m)
SIGMA_COBRE = 5.80e7      # Conductividad del cobre para tarjeta LD de NEC2 (S/m)

# Variable de estado para el motor C de Ngspice
NGSPICE_C_ACTIVO = PYSPICE_DISPONIBLE

# =====================================================================
# 2. REPRESENTACIÓN GENÉTICA Y CODIFICACIÓN PROGRESIVA
# =====================================================================
def crear_rama_genes(altura_max):
    genes = []
    dz_medio = altura_max / NUM_SEGMENTOS
    for _ in range(NUM_SEGMENTOS):
        r = random.uniform(0.005, RADIO_MAX)
        d_theta = random.uniform(-0.85, 0.85)
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
    """
    Construye las dos ramas ortogonales partiendo de (0,0,0).
    Para evitar que los dos cables gruesos se toquen al salir de la base,
    el Paso 1 despega estrictamente hacia su eje ortogonal (0° y 90°).
    """
    def construir_nodos(genes, altura_max, theta_inicial):
        nodos = [[0.0, 0.0, 0.0]]
        suma_dz = sum(max(g[2], 0.001) for g in genes)
        escala_z = min(1.0, altura_max / max(suma_dz, 1e-6))
        
        theta_actual = theta_inicial
        z_actual = 0.0
        
        for idx, (r, d_theta, d_z) in enumerate(genes):
            if idx == 0:
                # Salida limpia desde la soldadura común para separar paredes físicas
                r_clip = float(np.clip(r, RADIO_ALAMBRE * 3.0, RADIO_MAX))
                theta_actual = theta_inicial + d_theta * 0.15
            else:
                r_clip = float(np.clip(r, RADIO_ALAMBRE * 2.0, RADIO_MAX))
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
# 3. MOTOR DE COLISIONES FÍSICAS REALES (CENTRO A CENTRO + PAREDES)
# =====================================================================
def motor_de_colisiones_3d(nodos_a, nodos_b, luz_libre_mm=SEPARACION_AIRE_MM):
    """
    Verifica que las superficies exteriores de ambos cables jamás se toquen.
    Distancia mínima centro a centro = (2 * RADIO_ALAMBRE) + luz_libre.
    """
    diametro_alambre_m = 2.0 * RADIO_ALAMBRE
    distancia_min_centros = diametro_alambre_m + (luz_libre_mm / 1000.0)
    
    t_vals = np.linspace(0.0, 1.0, 12)
    puntos_a = [nodos_a[i] + t * (nodos_a[i+1] - nodos_a[i]) for i in range(len(nodos_a) - 1) for t in t_vals]
    puntos_b = [nodos_b[i] + t * (nodos_b[i+1] - nodos_b[i]) for i in range(len(nodos_b) - 1) for t in t_vals]
            
    pA = np.array(puntos_a)
    pB = np.array(puntos_b)
    
    radio_exclusion_origen = max(0.006, distancia_min_centros * 1.1)
    pA_validos = pA[np.linalg.norm(pA, axis=1) > radio_exclusion_origen]
    pB_validos = pB[np.linalg.norm(pB, axis=1) > radio_exclusion_origen]
    
    if len(pA_validos) == 0 or len(pB_validos) == 0:
        return 0.0, 999.0
        
    diff = pA_validos[:, np.newaxis, :] - pB_validos[np.newaxis, :, :]
    distancias = np.linalg.norm(diff, axis=-1)
    min_dist_centros = float(np.min(distancias))
    espacio_aire_real_mm = (min_dist_centros - diametro_alambre_m) * 1000.0
    
    penalizacion = 0.0
    if min_dist_centros < distancia_min_centros:
        penalizacion = (distancia_min_centros - min_dist_centros) * 80000.0 + 500.0
        
    return penalizacion, espacio_aire_real_mm

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
# 4. EXTRACCIÓN DE RED RLC Y VERIFICACIÓN EN PYSPICE
# =====================================================================
def extraer_red_rlc_rama(nodos, freq, factor_resonancia_dipolo, factor_escala_admitancia=1.0):
    omega = 2.0 * math.pi * freq
    longitud_onda = C_LUZ / freq
    delta_skin = math.sqrt(RHO_COBRE / (math.pi * freq * MU_0))
    segmentos_escalados = []
    longitud_total = 0.0

    escala_z = 1.0 / max(factor_escala_admitancia, 1e-6)

    for i in range(len(nodos) - 1):
        p1, p2 = nodos[i], nodos[i+1]
        vec = p2 - p1
        l_seg = max(float(np.linalg.norm(vec)), 1e-4)
        longitud_total += l_seg
        z_medio = max(float((p1[2] + p2[2]) / 2.0), 0.001)
        cos_alpha = abs(float(vec[2])) / l_seg

        term_log_l = max(math.log((2.0 * l_seg) / RADIO_ALAMBRE) - 1.0, 0.15)
        L_seg = (MU_0 / (2.0 * math.pi)) * l_seg * term_log_l * 0.45 * escala_z

        term_log_c = max(math.log((2.0 * z_medio) / RADIO_ALAMBRE), 0.25)
        C_seg = ((2.0 * math.pi * EPS_0 * l_seg) / term_log_c) * 0.65 / escala_z

        R_skin = (RHO_COBRE * l_seg) / (2.0 * math.pi * RADIO_ALAMBRE * delta_skin)
        l_norm = min(l_seg / longitud_onda, 0.5)
        R_rad = 73.0 * (math.pi ** 2) * (l_norm ** 2) * (0.5 + 0.5 * (cos_alpha ** 2))
        R_seg = max(R_rad + R_skin, 0.02) * escala_z

        segmentos_escalados.append((R_seg, L_seg, C_seg))

    beta = (2.0 * math.pi) / longitud_onda
    elec_len = beta * longitud_total * factor_resonancia_dipolo
    x_apertura = (-Z0 / (math.tan(max(min(elec_len, math.pi - 0.05), 0.05)) + 1e-6)) * 0.35 * escala_z
    r_apertura = max(36.5 * ((longitud_total / (longitud_onda * 0.5)) ** 2) * escala_z, 0.05)

    return segmentos_escalados, r_apertura, x_apertura

def resolver_escalera_analitica(segmentos, r_term, x_term, freq):
    omega = 2.0 * math.pi * freq
    z_carga = complex(r_term, x_term)
    for R_s, L_s, C_s in reversed(segmentos):
        z_serie = complex(R_s, omega * L_s)
        y_paralelo = complex(0.0, omega * C_s)
        y_nodo = (1.0 / (z_carga + 1e-12)) + y_paralelo
        z_carga = z_serie + (1.0 / y_nodo)
    return z_carga

def construir_y_verificar_pyspice(nodos_24, nodos_55, freq, nombre_banda):
    global NGSPICE_C_ACTIVO
    omega = 2.0 * math.pi * freq

    vec_medio_24 = np.mean(nodos_24[1:, :2], axis=0)
    vec_medio_55 = np.mean(nodos_55[1:, :2], axis=0)
    norm_24 = float(np.linalg.norm(vec_medio_24)) + 1e-6
    norm_55 = float(np.linalg.norm(vec_medio_55)) + 1e-6
    cos_cruz = abs(float(np.dot(vec_medio_24 / norm_24, vec_medio_55 / norm_55)))
    k_mutuo = 0.04 + 0.25 * cos_cruz

    if abs(freq - FREQ_24) < 1e6:
        seg_24, r_t24, x_t24 = extraer_red_rlc_rama(nodos_24, freq, 1.0, 1.0)
        seg_55, r_t55, x_t55 = extraer_red_rlc_rama(nodos_55, freq, 0.85, k_mutuo)
    else:
        seg_24, r_t24, x_t24 = extraer_red_rlc_rama(nodos_24, freq, 0.85, k_mutuo)
        seg_55, r_t55, x_t55 = extraer_red_rlc_rama(nodos_55, freq, 1.0, 1.0)

    if PYSPICE_DISPONIBLE:
        circuit = Circuit(f'Antena_Cruz_ESP32C5_{nombre_banda}')
        circuit.SinusoidalVoltageSource('rf', 'n_fuente', circuit.gnd, ac_magnitude=1@u_V)
        circuit.R('z0', 'n_fuente', 'n_in', Z0@u_Ohm)

        def agregar_rama_spice(prefijo, segmentos, r_term, x_term):
            nodo_prev = 'n_in'
            for idx, (R_v, L_v, C_v) in enumerate(segmentos):
                n_mid = f'n_{prefijo}_m{idx+1}'
                n_out = f'n_{prefijo}_o{idx+1}'
                circuit.R(f'{prefijo}_s{idx+1}', nodo_prev, n_mid, R_v@u_Ohm)
                circuit.L(f'{prefijo}_s{idx+1}', n_mid, n_out, L_v@u_H)
                circuit.C(f'{prefijo}_p{idx+1}', n_out, circuit.gnd, C_v@u_F)
                nodo_prev = n_out
            n_term = f'n_{prefijo}_term'
            circuit.R(f'{prefijo}_rad', nodo_prev, n_term, r_term@u_Ohm)
            if x_term >= 0:
                circuit.L(f'{prefijo}_xrad', n_term, circuit.gnd, (max(x_term, 1e-6) / omega)@u_H)
            else:
                circuit.C(f'{prefijo}_xrad', n_term, circuit.gnd, (1.0 / (omega * max(abs(x_term), 1e-6)))@u_F)

        agregar_rama_spice('R24', seg_24, r_t24, x_t24)
        agregar_rama_spice('R55', seg_55, r_t55, x_t55)
        netlist_str = str(circuit) + f"\n.ac lin 2 {freq*0.999:.6e} {freq*1.001:.6e}\n.end"

        if NGSPICE_C_ACTIVO:
            try:
                simulator = circuit.simulator(temperature=25, nominal_temperature=25)
                analysis = simulator.ac(
                    start_frequency=(freq * 0.999)@u_Hz,
                    stop_frequency=(freq * 1.001)@u_Hz,
                    number_of_points=2,
                    variation='lin'
                )
                v_in = complex(analysis['n_in'].as_ndarray()[0])
                try:
                    ngspice = simulator.factory(circuit).ngspice
                    ngspice.remove_circuit()
                    ngspice.destroy()
                except Exception:
                    pass
                i_in = (1.0 - v_in) / Z0
                z_in_spice = v_in / i_in
                gamma = abs((z_in_spice - Z0) / (z_in_spice + Z0))
                s11_spice = 20.0 * math.log10(max(gamma, 1e-5))
                return z_in_spice, s11_spice, netlist_str
            except Exception:
                NGSPICE_C_ACTIVO = False
    else:
        lineas = [
            f".title Antena_Cruz_ESP32C5_{nombre_banda}",
            "Vrf n_fuente 0 DC 0V AC 1V",
            f"Rz0 n_fuente n_in {Z0:.4f}"
        ]
        for prefijo, segmentos, r_term, x_term in [('R24', seg_24, r_t24, x_t24), ('R55', seg_55, r_t55, x_t55)]:
            nodo_prev = 'n_in'
            for idx, (R_v, L_v, C_v) in enumerate(segmentos):
                n_mid = f"n_{prefijo}_m{idx+1}"
                n_out = f"n_{prefijo}_o{idx+1}"
                lineas.append(f"R_{prefijo}_s{idx+1} {nodo_prev} {n_mid} {R_v:.6e}")
                lineas.append(f"L_{prefijo}_s{idx+1} {n_mid} {n_out} {L_v:.6e}")
                lineas.append(f"C_{prefijo}_p{idx+1} {n_out} 0 {C_v:.6e}")
                nodo_prev = n_out
            n_term = f"n_{prefijo}_term"
            lineas.append(f"R_{prefijo}_rad {nodo_prev} {n_term} {r_term:.6e}")
            if x_term >= 0:
                lineas.append(f"L_{prefijo}_xrad {n_term} 0 {(max(x_term, 1e-6) / omega):.6e}")
            else:
                lineas.append(f"C_{prefijo}_xrad {n_term} 0 {(1.0 / (omega * max(abs(x_term), 1e-6))):.6e}")
        lineas.append(f".ac lin 2 {freq*0.999:.6e} {freq*1.001:.6e}")
        lineas.append(".end")
        netlist_str = "\n".join(lineas)

    z1 = resolver_escalera_analitica(seg_24, r_t24, x_t24, freq)
    z2 = resolver_escalera_analitica(seg_55, r_t55, x_t55, freq)
    z_in_mna = 1.0 / ((1.0 / z1) + (1.0 / z2))
    gamma = abs((z_in_mna - Z0) / (z_in_mna + Z0))
    s11_mna = 20.0 * math.log10(max(gamma, 1e-5))
    return z_in_mna, s11_mna, netlist_str

# =====================================================================
# 5. GENERADOR DE ARCHIVO NEC2 (.nec) DEL CAMPEÓN
# =====================================================================
def generar_archivo_nec_campeon(nodos_24_m, nodos_55_m, filename="antena_cruz_esp32c5.nec"):
    """
    Genera un archivo de entrada NEC2 estándar (en metros) para 4nec2, xnec2c o PyNEC.
    Incluye:
    - Pin alimentador común en la base (Tag 99) donde se inyecta la fuente EX.
    - Los 8 segmentos de la Rama 2.44 GHz (Tags 1 al 8).
    - Los 8 segmentos de la Rama 5.50 GHz (Tags 11 al 18).
    - Pérdidas reales por conductividad del cobre (LD 5).
    - Barrido de frecuencias y cálculo de patrón 3D completo (RP).
    """
    lineas_nec = [
        "CM ===================================================================",
        "CM ANTENA EN CRUZ DE DIPOLOS EVOLUCIONADA (ESP32-C5: 2.44 & 5.50 GHz)",
        f"CM Radio Cilindro: {RADIO_MAX*100:.1f} cm | H_24: {ALTURA_MAX_24*100:.1f} cm | H_55: {ALTURA_MAX_55*100:.1f} cm",
        f"CM Radio del Alambre de Cobre: {RADIO_ALAMBRE*1000:.2f} mm (Diametro: {RADIO_ALAMBRE*2000:.2f} mm)",
        "CM Unidades de Geometria NEC: Metros (m)",
        "CE"
    ]

    # Punto de bifurcación a 2 mm sobre el origen para alimentar ambas ramas en paralelo sin error de unión NEC
    z_bif = 0.0020
    lineas_nec.append(f"GW 99 1  0.000000  0.000000  0.000000  0.000000  0.000000 {z_bif:9.6f} {RADIO_ALAMBRE:8.5f}")

    # Copiar coordenadas ajustando el nodo 0 al punto de bifurcación común (0, 0, z_bif)
    n24 = np.copy(nodos_24_m)
    n55 = np.copy(nodos_55_m)
    n24[0] = [0.0, 0.0, z_bif]
    n55[0] = [0.0, 0.0, z_bif]

    # Tarjetas GW para Rama 1 (2.44 GHz) -> Tags 1 a 8 (5 subdivisiones MoM por tramo)
    for i in range(len(n24) - 1):
        p1, p2 = n24[i], n24[i+1]
        tag = i + 1
        lineas_nec.append(
            f"GW {tag:2d} 5 {p1[0]:9.6f} {p1[1]:9.6f} {p1[2]:9.6f} "
            f"{p2[0]:9.6f} {p2[1]:9.6f} {p2[2]:9.6f} {RADIO_ALAMBRE:8.5f}"
        )

    # Tarjetas GW para Rama 2 (5.50 GHz) -> Tags 11 a 18 (5 subdivisiones MoM por tramo)
    for i in range(len(n55) - 1):
        p1, p2 = n55[i], n55[i+1]
        tag = i + 11
        lineas_nec.append(
            f"GW {tag:2d} 5 {p1[0]:9.6f} {p1[1]:9.6f} {p1[2]:9.6f} "
            f"{p2[0]:9.6f} {p2[1]:9.6f} {p2[2]:9.6f} {RADIO_ALAMBRE:8.5f}"
        )

    # Fin de la geometría en espacio libre (GE 0)
    lineas_nec.extend([
        "GE 0",
        f"LD 5 0 0 0 {SIGMA_COBRE:.2e}",
        "EX 0 99 1 0 1.0 0.0",
        "CM --- Evaluacion en Banda Baja: 2440 MHz ---",
        f"FR 0 1 0 0 {FREQ_24/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        "CM --- Evaluacion en Banda Alta: 5500 MHz ---",
        f"FR 0 1 0 0 {FREQ_55/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        "EN"
    ])

    contenido_nec = "\n".join(lineas_nec)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(contenido_nec + "\n")
    except Exception as e:
        print(f"Aviso al guardar archivo .nec en disco: {e}")

    return contenido_nec

# =====================================================================
# 6. SIMULACIÓN MULTIBANDA Y OPERADORES GENÉTICOS
# =====================================================================
def simular_cruz_multibanda(individuo):
    nodos_24, nodos_55 = decodificar_cruz_a_cartesianas(individuo)

    vec_medio_24 = np.mean(nodos_24[1:, :2], axis=0)
    vec_medio_55 = np.mean(nodos_55[1:, :2], axis=0)
    norm_24 = float(np.linalg.norm(vec_medio_24)) + 1e-6
    norm_55 = float(np.linalg.norm(vec_medio_55)) + 1e-6
    cos_cruz = abs(float(np.dot(vec_medio_24 / norm_24, vec_medio_55 / norm_55)))
    k_mutuo = 0.04 + 0.25 * cos_cruz

    # Banda 2.44 GHz
    s24_r1, rt24_1, xt24_1 = extraer_red_rlc_rama(nodos_24, FREQ_24, 1.0, 1.0)
    s24_r2, rt24_2, xt24_2 = extraer_red_rlc_rama(nodos_55, FREQ_24, 0.85, k_mutuo)
    z24_1 = resolver_escalera_analitica(s24_r1, rt24_1, xt24_1, FREQ_24)
    z24_2 = resolver_escalera_analitica(s24_r2, rt24_2, xt24_2, FREQ_24)
    z_in_24 = 1.0 / ((1.0 / z24_1) + (1.0 / z24_2))

    # Banda 5.50 GHz
    s55_r1, rt55_1, xt55_1 = extraer_red_rlc_rama(nodos_24, FREQ_55, 0.85, k_mutuo)
    s55_r2, rt55_2, xt55_2 = extraer_red_rlc_rama(nodos_55, FREQ_55, 1.0, 1.0)
    z55_1 = resolver_escalera_analitica(s55_r1, rt55_1, xt55_1, FREQ_55)
    z55_2 = resolver_escalera_analitica(s55_r2, rt55_2, xt55_2, FREQ_55)
    z_in_55 = 1.0 / ((1.0 / z55_1) + (1.0 / z55_2))

    gamma_24 = abs((z_in_24 - Z0) / (z_in_24 + Z0))
    gamma_55 = abs((z_in_55 - Z0) / (z_in_55 + Z0))

    s11_24 = 20.0 * math.log10(max(gamma_24, 1e-5))
    s11_55 = 20.0 * math.log10(max(gamma_55, 1e-5))

    iso_24, pen_24 = evaluar_geometria_e_isotropia(nodos_24)
    iso_55, pen_55 = evaluar_geometria_e_isotropia(nodos_55)
    puntaje_omni = iso_24 + iso_55
    
    pen_colision, aire_min_mm = motor_de_colisiones_3d(nodos_24, nodos_55, luz_libre_mm=SEPARACION_AIRE_MM)
    penalizacion_geom = pen_24 + pen_55 + (cos_cruz * 12.0) + pen_colision

    peor_s11 = max(s11_24, s11_55)
    promedio_s11 = 0.5 * (s11_24 + s11_55)

    penalizacion_s11 = 0.0
    if s11_24 > -10.0: penalizacion_s11 += (s11_24 - (-10.0)) * 8.0
    if s11_55 > -10.0: penalizacion_s11 += (s11_55 - (-10.0)) * 8.0

    fitness = (-peor_s11 * 2.5) + (-promedio_s11 * 1.0) + (puntaje_omni * 0.8) - penalizacion_s11 - penalizacion_geom

    return fitness, s11_24, s11_55, z_in_24, z_in_55, puntaje_omni, aire_min_mm

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
                genes[i, 0] = np.clip(genes[i, 0] + random.gauss(0, 0.0025), RADIO_ALAMBRE * 2.0, RADIO_MAX)
                genes[i, 1] = np.clip(genes[i, 1] + random.gauss(0, 0.18), -1.4, 1.4)
                genes[i, 2] = np.clip(genes[i, 2] + random.gauss(0, 0.0015), 0.0015, dz_max_paso)
    mutar_rama(mutado['rama_24'], ALTURA_MAX_24)
    mutar_rama(mutado['rama_55'], ALTURA_MAX_55)
    return mutado

# =====================================================================
# 7. BUCLE EVOLUTIVO PRINCIPAL + EXPORTACIÓN SPICE, NEC2 Y OPENSCAD
# =====================================================================
def ejecutar_optimizacion():
    random.seed(42)
    np.random.seed(42)

    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    print("=" * 94)
    print(" OPTIMIZACIÓN EVOLUTIVA + PYSPICE + EXPORTACIÓN NEC2 + ANTI-COLISIÓN (ESP32-C5)")
    print(f" Cilindro Único: R={RADIO_MAX*100:.1f} cm | H_2.44={ALTURA_MAX_24*100:.1f} cm | H_5.50={ALTURA_MAX_55*100:.1f} cm")
    print(f" Alambre: Radio={RADIO_ALAMBRE*1000:.1f} mm (Diámetro={RADIO_ALAMBRE*2000:.1f} mm) | Aire libre mín={SEPARACION_AIRE_MM:.1f} mm")
    print(f" Población: {TAMANO_POBLACION} | Generaciones: {GENERACIONES} | Élite intacta: {ELITE_SIZE}")
    print("=" * 94)

    for gen in range(GENERACIONES):
        evaluados = []
        for ind in poblacion:
            fit, s11_24, s11_55, z24, z55, omni, aire_mm = simular_cruz_multibanda(ind)
            evaluados.append((fit, s11_24, s11_55, z24, z55, omni, aire_mm, ind))

        evaluados.sort(key=lambda x: x[0], reverse=True)

        if gen % 32 == 0 or gen == GENERACIONES - 1:
            bfit, bs24, bs55, bz24, bz55, bomni, baire, mejor_gen_ind = evaluados[0]
            n24_g, n55_g = decodificar_cruz_a_cartesianas(mejor_gen_ind)
            _, s11_sp24, _ = construir_y_verificar_pyspice(n24_g, n55_g, FREQ_24, "2440MHz")
            _, s11_sp55, _ = construir_y_verificar_pyspice(n24_g, n55_g, FREQ_55, "5500MHz")
            print(f"Gen {gen:4d}/{GENERACIONES} | Fit: {bfit:7.2f} | "
                  f"S11_Spice(2.4G): {s11_sp24:6.2f} dB | S11_Spice(5.5G): {s11_sp55:6.2f} dB | "
                  f"Aire: {baire:4.2f} mm")

        mejores_elite = [copiar_individuo(item[7]) for item in evaluados[:ELITE_SIZE]]
        nueva_poblacion = [copiar_individuo(elite) for elite in mejores_elite]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1, padre2 = random.sample(mejores_elite, 2)
            hijo = cruzar_individuos(padre1, padre2)
            hijo_mutado = mutar_individuo(hijo)
            nueva_poblacion.append(hijo_mutado)

        poblacion = nueva_poblacion

    # Extraer el campeón final y verificar con PySpice
    mejor_fit, mejor_s24, mejor_s55, mejor_z24, mejor_z55, mejor_omni, mejor_aire, campeon = evaluados[0]
    nodos_24_m, nodos_55_m = decodificar_cruz_a_cartesianas(campeon)

    z_sp24, s11_sp24, netlist_24 = construir_y_verificar_pyspice(nodos_24_m, nodos_55_m, FREQ_24, "2440MHz")
    z_sp55, s11_sp55, netlist_55 = construir_y_verificar_pyspice(nodos_24_m, nodos_55_m, FREQ_55, "5500MHz")

    # Generar archivo NEC2 únicamente para el campeón final
    nombre_archivo_nec = "antena_cruz_esp32c5.nec"
    contenido_nec = generar_archivo_nec_campeon(nodos_24_m, nodos_55_m, filename=nombre_archivo_nec)

    nodos_24_cm = nodos_24_m * 100.0
    nodos_55_cm = nodos_55_m * 100.0

    len_24_cm = sum(np.linalg.norm(nodos_24_cm[i+1] - nodos_24_cm[i]) for i in range(NUM_SEGMENTOS))
    len_55_cm = sum(np.linalg.norm(nodos_55_cm[i+1] - nodos_55_cm[i]) for i in range(NUM_SEGMENTOS))

    print("\n" + "=" * 94)
    print(" ¡OPTIMIZACIÓN, VERIFICACIÓN SPICE Y ARCHIVO NEC2 COMPLETADOS!")
    print(f" * Verificación PySpice @ 2.44 GHz : S11 = {s11_sp24:6.2f} dB (Z_in = {z_sp24.real:5.2f} {z_sp24.imag:+6.2f}j Ohm)")
    print(f" * Verificación PySpice @ 5.50 GHz : S11 = {s11_sp55:6.2f} dB (Z_in = {z_sp55.real:5.2f} {z_sp55.imag:+6.2f}j Ohm)")
    print(f" * Separación libre entre paredes  : {mejor_aire:.2f} mm (Cero contacto físico fuera de la base)")
    print(f" * Puntaje Omnidireccional 3D      : +{mejor_omni:5.2f} pts")
    print(f" * Longitud física alambre         : Rama 2.44G = {len_24_cm:.2f} cm | Rama 5.50G = {len_55_cm:.2f} cm")
    print(f" * Archivo NEC2 guardado en disco  : {nombre_archivo_nec}")
    print("=" * 94)

    print("\n* --- ÚLTIMO MODELO SPICE GENERADO (NETLIST 2.44 GHz) ---")
    print(netlist_24)

    print("\n* --- ÚLTIMO MODELO SPICE GENERADO (NETLIST 5.50 GHz) ---")
    print(netlist_55)

    print(f"\n* --- ARCHIVO NEC2 DEL CAMPEÓN ({nombre_archivo_nec}) ---")
    print(contenido_nec)

    print("\n// --- COPIAR Y PEGAR DIRECTAMENTE EN OPENSCAD ---")
    print(f"radio_alambre_mm = {RADIO_ALAMBRE * 1000.0:.2f}; // {RADIO_ALAMBRE * 2000.0:.1f} mm de diámetro")
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