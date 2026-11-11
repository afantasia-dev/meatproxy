import numpy as np
import random
import math

# =====================================================================
# CARGA DE PYNEC (necpp) CON PROTECCIÓN CONTRA SEGMENTATION FAULT
# =====================================================================
PYNEC_DISPONIBLE = False
try:
    import necpp
    PYNEC_DISPONIBLE = True
except ImportError:
    print("Aviso: Librería 'necpp' (PyNEC) no detectada.")

USAR_PYNEC_EN_BUCLE = False

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y ELÉCTRICOS (100% ENFOCADO EN 5.50 GHz)
# =====================================================================
RADIO_MAX = 0.040         # 4.0 cm de radio máximo del cilindro (0.04 m)
ALTURA_MAX = 0.070        # 7.0 cm de altura máxima del cilindro (0.07 m)

# Para 5.50 GHz (lambda = 5.45 cm), permitimos desde ~1.25 lambda (6.8 cm) hasta ~2.5 lambda (13.6 cm)
LONGITUD_MIN_M = 0.068    # 6.8 cm longitud mínima estirada
LONGITUD_MAX_M = 0.136    # 13.6 cm longitud máxima estirada

RADIO_ALAMBRE = 0.0020    # Radio 2.0 mm (Alambre grueso de 4.0 mm de diámetro)
SEPARACION_AIRE_MM = 2.5  # Espacio libre mínimo contra auto-intersección (mm)
NUM_SEGMENTOS = 10        # 10 pasos 3D progresivos

FREQ_55 = 5.50e9          # ÚNICA BANDA OBJETIVO DE OPTIMIZACIÓN (5.50 GHz)
FREQ_24_INFO = 2.44e9     # Solo para referencia en el archivo .nec final (ignorada en el algoritmo)
Z0 = 50.0                 # Impedancia puerto RF ESP32-C5 (50 Ohm)

TAMANO_POBLACION = 256 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 1024
GENERACIONES = 512 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 1024
ELITE_SIZE = 32           # Élite compacta para evitar estancamiento por clones
TASA_MUTACION = 0.45      # Alta variación genética (45%)

# Constantes físicas
C_LUZ = 299792458.0
EPS_0 = 8.854187817e-12
MU_0 = 4.0 * math.pi * 1e-7
RHO_COBRE = 1.68e-8
SIGMA_COBRE = 5.80e7      # Conductividad del cobre para tarjeta LD de NEC2 (S/m)

# =====================================================================
# 2. CODIFICACIÓN DE HÉLICE PROGRESIVA (EVITA PATRONES ABOLLADOS)
# =====================================================================
def crear_individuo():
    genes = []
    dz_medio = ALTURA_MAX / NUM_SEGMENTOS
    for _ in range(NUM_SEGMENTOS):
        r = random.uniform(0.008, RADIO_MAX)
        d_theta = random.uniform(0.25, 1.35)
        d_z = random.uniform(dz_medio * 0.5, dz_medio * 1.45)
        genes.append([r, d_theta, d_z])
    return np.array(genes, dtype=float)

def copiar_individuo(ind):
    return np.copy(ind)

def decodificar_alambre_a_cartesianas(genes):
    """
    Convierte los genes en coordenadas 3D (X, Y, Z) partiendo del origen (0,0,0)[cite: 1].
    """
    nodos = [[0.0, 0.0, 0.0]]
    suma_dz = sum(max(g[2], 0.0025) for g in genes)
    escala_z = min(1.0, ALTURA_MAX / max(suma_dz, 1e-6))

    theta_actual = 0.0
    z_actual = 0.0
    r_prev = RADIO_ALAMBRE * 3.5

    for idx, (r, d_theta, d_z) in enumerate(genes):
        r_target = float(np.clip(r, RADIO_ALAMBRE * 3.0, RADIO_MAX))
        # Respuesta radial ágil (85%) para permitir explorar geometrías con mayor libertad
        r_clip = 0.15 * r_prev + 0.85 * r_target
        r_prev = r_clip

        theta_actual += d_theta
        paso_z = max(d_z * escala_z, 0.0035)
        z_actual = min(ALTURA_MAX, z_actual + paso_z)

        x = r_clip * math.cos(theta_actual)
        y = r_clip * math.sin(theta_actual)
        nodos.append([x, y, float(z_actual)])

    return np.array(nodos, dtype=float)

# =====================================================================
# 3. AUTO-COLISIÓN Y MODELO DE RADIACIÓN 3D EXCLUSIVO PARA 5.50 GHz
# =====================================================================
def motor_autocolision_alambre(nodos, luz_libre_mm=SEPARACION_AIRE_MM):
    diametro_m = 2.0 * RADIO_ALAMBRE
    dist_min_req = diametro_m + (luz_libre_mm / 1000.0)
    t_vals = np.linspace(0.0, 1.0, 10)

    segmentos_pts = []
    for i in range(len(nodos) - 1):
        pts = np.array([nodos[i] + t * (nodos[i+1] - nodos[i]) for t in t_vals])
        segmentos_pts.append(pts)

    min_dist_centros = 999.0
    penalizacion = 0.0

    for i in range(len(nodos) - 1):
        l_tramo = float(np.linalg.norm(nodos[i+1] - nodos[i]))
        if l_tramo < diametro_m * 1.5:
            penalizacion += 5000.0

    for i in range(len(segmentos_pts)):
        for j in range(i + 2, len(segmentos_pts)):
            diff = segmentos_pts[i][:, np.newaxis, :] - segmentos_pts[j][np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            d_ij = float(np.min(dists))
            if d_ij < min_dist_centros:
                min_dist_centros = d_ij
            if d_ij < dist_min_req:
                penalizacion += (dist_min_req - d_ij) * 80000.0 + 300.0

    aire_libre_mm = (min_dist_centros - diametro_m) * 1000.0
    return penalizacion, aire_libre_mm

def cmath_exp_j(angulo):
    return complex(math.cos(angulo), math.sin(angulo))

def evaluar_patron_3d_55ghz(nodos):
    """
    Evalúa el patrón 3D exclusivamente en 5.50 GHz (ignorando 2.44 GHz):
    - Muestrea 3 anillos en el hemisferio +Z (theta = 30°, 60°, 90°) + el eje +Z puro (0°).
    - Minimiza el rizado X-Y (abolladuras) y el spread 3D global en 5.50 GHz.
    """
    vectores = []
    longitudes = []
    centros = []
    pos_acumulada = []
    penalizacion_geom = 0.0
    l_acum = 0.0

    for i in range(len(nodos) - 1):
        vec = nodos[i+1] - nodos[i]
        l_seg = float(np.linalg.norm(vec))
        if l_seg < 1e-5:
            penalizacion_geom += 20.0
            continue
        vectores.append(vec / l_seg)
        longitudes.append(l_seg)
        centros.append(0.5 * (nodos[i] + nodos[i+1]))
        pos_acumulada.append(l_acum + 0.5 * l_seg)
        l_acum += l_seg

    longitud_total = l_acum

    if longitud_total < LONGITUD_MIN_M:
        penalizacion_geom += (LONGITUD_MIN_M - longitud_total) * 1500.0
    elif longitud_total > LONGITUD_MAX_M:
        penalizacion_geom += (longitud_total - LONGITUD_MAX_M) * 1500.0

    # Suavidad entre segmentos para evitar lóbulos rotos a 5.50 GHz
    for i in range(len(vectores) - 1):
        cos_ang = float(np.dot(vectores[i], vectores[i+1]))
        if cos_ang < 0.25:
            penalizacion_geom += (0.25 - cos_ang) * 30.0

    # Centrado de masa en X-Y para simetría azimutal
    centro_masa_xy = np.mean(nodos[1:, :2], axis=0)
    penalizacion_geom += float(np.linalg.norm(centro_masa_xy)) * 550.0

    k = (2.0 * math.pi * FREQ_55) / C_LUZ
    k_eff = k * 0.88

    def calcular_potencia_dir_55(theta, phi):
        r_hat = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
        e_vec = np.zeros(3, dtype=complex)
        for u_vec, l_s, c_pt, s_pos in zip(vectores, longitudes, centros, pos_acumulada):
            dist_punta = longitud_total - s_pos
            i_mag = math.sin(k_eff * dist_punta) + 0.25j * math.cos(k_eff * dist_punta)
            fase_espacial = cmath_exp_j(k * float(np.dot(r_hat, c_pt)))
            trans = u_vec - r_hat * float(np.dot(u_vec, r_hat))
            e_vec += trans * l_s * i_mag * fase_espacial
        return float(np.sum(np.abs(e_vec) ** 2)) + 1e-9

    # Muestreo denso del domo +Z exclusivamente a 5.50 GHz:
    # Anillo horizonte (90°), Anillo medio (60°), Anillo alto (30°) y Cúspide +Z (0°)
    phis = np.linspace(0, 2.0 * math.pi, 12, endpoint=False)
    p_xy_90 = [10.0 * math.log10(calcular_potencia_dir_55(math.pi * 0.50, phi)) for phi in phis]
    p_xy_60 = [10.0 * math.log10(calcular_potencia_dir_55(math.pi * 0.33, phi)) for phi in phis]
    p_xy_30 = [10.0 * math.log10(calcular_potencia_dir_55(math.pi * 0.16, phi)) for phi in phis]
    p_z_top = 10.0 * math.log10(calcular_potencia_dir_55(0.0, 0.0))

    # Rizado azimutal X-Y (qué tan circular y libre de abolladuras es la dona de 5.5 GHz)
    ripple_90 = (max(p_xy_90) - min(p_xy_90)) + 2.0 * float(np.std(p_xy_90))
    ripple_60 = (max(p_xy_60) - min(p_xy_60)) + 2.0 * float(np.std(p_xy_60))
    ripple_30 = (max(p_xy_30) - min(p_xy_30)) + 2.0 * float(np.std(p_xy_30))
    rizado_xy_55 = (ripple_90 + ripple_60 + ripple_30) / 3.0

    # Variación global 3D en todo el hemisferio +Z (XY + Z)
    todos_pts = p_xy_90 + p_xy_60 + p_xy_30 + [p_z_top]
    spread_3d_55 = max(todos_pts) - min(todos_pts)

    # Equilibrio entre ganancia en +Z y ganancia en el horizonte X-Y
    media_xy = float(np.mean(p_xy_90))
    deficit_z_55 = abs(p_z_top - media_xy) * 1.8

    return spread_3d_55, rizado_xy_55, deficit_z_55, penalizacion_geom, longitud_total

# =====================================================================
# 4. MODELO DE IMPEDANCIA (UN SOLO ALAMBRE)
# =====================================================================
def calcular_z_in_un_alambre(nodos, freq):
    omega = 2.0 * math.pi * freq
    longitud_onda = C_LUZ / freq
    delta_skin = math.sqrt(RHO_COBRE / (math.pi * freq * MU_0))
    segmentos = []
    longitud_total = 0.0

    for i in range(len(nodos) - 1):
        p1, p2 = nodos[i], nodos[i+1]
        vec = p2 - p1
        l_seg = max(float(np.linalg.norm(vec)), 1e-4)
        longitud_total += l_seg
        z_medio = max(float((p1[2] + p2[2]) / 2.0), 0.001)
        cos_alpha = abs(float(vec[2])) / l_seg

        term_log_l = max(math.log((2.0 * l_seg) / RADIO_ALAMBRE) - 1.0, 0.15)
        L_seg = (MU_0 / (2.0 * math.pi)) * l_seg * term_log_l * 0.42

        term_log_c = max(math.log((2.0 * z_medio) / RADIO_ALAMBRE), 0.25)
        C_seg = ((2.0 * math.pi * EPS_0 * l_seg) / term_log_c) * 0.68

        R_skin = (RHO_COBRE * l_seg) / (2.0 * math.pi * RADIO_ALAMBRE * delta_skin)
        l_norm = min(l_seg / longitud_onda, 0.5)
        R_rad = 73.0 * (math.pi ** 2) * (l_norm ** 2) * (0.5 + 0.5 * (cos_alpha ** 2))
        R_seg = max(R_rad + R_skin, 0.02)

        segmentos.append((R_seg, L_seg, C_seg))

    beta = (2.0 * math.pi) / longitud_onda
    elec_len = beta * longitud_total * 0.82
    x_apertura = (-Z0 / (math.tan(elec_len) + 1e-6)) * 0.30
    x_apertura = float(np.clip(x_apertura, -250.0, 250.0))
    r_apertura = 38.0 + 12.0 * abs(math.sin(elec_len))

    z_carga = complex(r_apertura, x_apertura)
    for R_s, L_s, C_s in reversed(segmentos):
        z_serie = complex(R_s, omega * L_s)
        y_paralelo = complex(0.0, omega * C_s)
        y_nodo = (1.0 / (z_carga + 1e-12)) + y_paralelo
        z_carga = z_serie + (1.0 / y_nodo)

    return z_carga

# =====================================================================
# 5. SIMULADOR PYNEC (necpp) BLINDADO CONTRA SEGFAULT + EXPORTADOR .NEC
# =====================================================================
def calcular_subsegmentos_seguros(l_tramo):
    longitud_min_subseg = RADIO_ALAMBRE * 2.2
    n_max = int(l_tramo // longitud_min_subseg)
    return max(1, min(5, n_max))

def simular_un_alambre_pynec(nodos_m, freq_hz):
    if not PYNEC_DISPONIBLE:
        return None

    for i in range(len(nodos_m) - 1):
        l_tramo = float(np.linalg.norm(nodos_m[i+1] - nodos_m[i]))
        if l_tramo < RADIO_ALAMBRE * 2.1:
            return None

    nec = necpp.nec_create()
    try:
        for i in range(len(nodos_m) - 1):
            p1, p2 = nodos_m[i], nodos_m[i+1]
            l_tramo = float(np.linalg.norm(p2 - p1))
            n_segs = calcular_subsegmentos_seguros(l_tramo)
            necpp.nec_wire(nec, i + 1, n_segs,
                           float(p1[0]), float(p1[1]), float(p1[2]),
                           float(p2[0]), float(p2[1]), float(p2[2]),
                           RADIO_ALAMBRE, 1.0, 1.0)

        necpp.nec_geometry_complete(nec, 0)
        necpp.nec_ld_card(nec, 5, 0, 0, 0, SIGMA_COBRE, 0.0, 0.0)
        necpp.nec_fr_card(nec, 0, 1, freq_hz / 1e6, 0.0)
        necpp.nec_ex_card(nec, 0, 1, 1, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        necpp.nec_rp_card(nec, 0, 19, 37, 0, 5, 0, 0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0)

        z_real = float(necpp.nec_impedance_real(nec, 0))
        z_imag = float(necpp.nec_impedance_imag(nec, 0))
        z_in_nec = complex(z_real, z_imag)

        gamma = abs((z_in_nec - Z0) / (z_in_nec + Z0))
        s11_nec = 20.0 * math.log10(max(gamma, 1e-5))

        gain_max = float(necpp.nec_gain_max(nec, 0))
        gain_min = float(necpp.nec_gain_min(nec, 0))
        gain_sd = float(necpp.nec_gain_sd(nec, 0))
        delta_dbi = gain_max - max(gain_min, -40.0)

        necpp.nec_delete(nec)
        return z_in_nec, s11_nec, gain_max, gain_min, delta_dbi, gain_sd
    except Exception:
        try:
            necpp.nec_delete(nec)
        except Exception:
            pass
        return None

def generar_archivo_nec_campeon(nodos_m, filename="antena_55ghz_pura_esp32c5.nec"):
    lineas_nec = [
        "CM ===================================================================",
        "CM ANTENA MONORRAMA OPTIMIZADA 100% PARA 5.50 GHz (ESP32-C5)",
        f"CM Cilindro: R={RADIO_MAX*100:.1f} cm | H={ALTURA_MAX*100:.1f} cm | Alambre D={RADIO_ALAMBRE*2000:.1f} mm",
        "CM Unidades NEC: Metros (m)",
        "CE"
    ]

    for i in range(len(nodos_m) - 1):
        p1, p2 = nodos_m[i], nodos_m[i+1]
        tag = i + 1
        l_tramo = float(np.linalg.norm(p2 - p1))
        n_segs = calcular_subsegmentos_seguros(l_tramo)
        lineas_nec.append(
            f"GW {tag:2d} {n_segs} {p1[0]:9.6f} {p1[1]:9.6f} {p1[2]:9.6f} "
            f"{p2[0]:9.6f} {p2[1]:9.6f} {p2[2]:9.6f} {RADIO_ALAMBRE:8.5f}"
        )

    lineas_nec.extend([
        "GE 0",
        f"LD 5 0 0 0 {SIGMA_COBRE:.2e}",
        "EX 0 1 1 0 1.0 0.0",
        f"FR 0 1 0 0 {FREQ_55/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        f"FR 0 1 0 0 {FREQ_24_INFO/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        "EN"
    ])

    contenido = "\n".join(lineas_nec)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(contenido + "\n")
    except Exception as e:
        print(f"Aviso al guardar archivo .nec: {e}")
    return contenido

# =====================================================================
# 6. FUNCIÓN DE APTITUD (100% ENFOCADA EN 5.50 GHz, IGNORA 2.44 GHz)
# =====================================================================
def evaluar_individuo(individuo):
    nodos = decodificar_alambre_a_cartesianas(individuo)

    pen_col, aire_mm = motor_autocolision_alambre(nodos, luz_libre_mm=SEPARACION_AIRE_MM)
    spread_3d_55, ripple_xy_55, deficit_z_55, pen_geom, len_total_m = evaluar_patron_3d_55ghz(nodos)

    if PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE and pen_col == 0.0:
        res55 = simular_un_alambre_pynec(nodos, FREQ_55)
    else:
        res55 = None

    if res55 is not None:
        z_in_55, s11_55, gmax55, gmin55, delta_dbi_55, sd55 = res55
        varianza_dbi_55 = delta_dbi_55 + 2.5 * sd55 + ripple_xy_55
    else:
        z_in_55 = calcular_z_in_un_alambre(nodos, FREQ_55)
        gamma_55 = abs((z_in_55 - Z0) / (z_in_55 + Z0))
        s11_55 = 20.0 * math.log10(max(gamma_55, 1e-5))
        varianza_dbi_55 = spread_3d_55 + (ripple_xy_55 * 1.8) + deficit_z_55

    penalizacion_s11 = 0.0
    if s11_55 > -12.0:
        penalizacion_s11 += (s11_55 - (-12.0)) * 12.0

    # El Fitness evalúa ÚNICAMENTE el desempeño en 5.50 GHz:
    # Topamos el premio de S11 en -45 dB para que una vez alcanzado un excelente acoplamiento
    # dedique todo el esfuerzo evolutivo a alisar la esfera (bajar varianza_dbi_55)
    s11_efectivo = max(s11_55, -45.0)
    fitness = (-s11_efectivo * 2.8) - (varianza_dbi_55 * 4.2) - penalizacion_s11 - pen_geom - pen_col

    return fitness, s11_55, z_in_55, spread_3d_55, ripple_xy_55, aire_mm, len_total_m

# =====================================================================
# 7. OPERADORES GENÉTICOS CON ALTA VARIACIÓN Y ANTI-ESTANCAMIENTO
# =====================================================================
def cruzar_individuos(padre1, padre2):
    hijo = crear_individuo()
    for i in range(NUM_SEGMENTOS):
        if random.random() < 0.5:
            hijo[i] = np.copy(padre1[i])
        else:
            hijo[i] = np.copy(padre2[i])
    return hijo

def mutar_individuo(individuo):
    mutado = copiar_individuo(individuo)
    dz_max_paso = (ALTURA_MAX / NUM_SEGMENTOS) * 1.8
    for i in range(NUM_SEGMENTOS):
        if random.random() < TASA_MUTACION:
            # 15% de probabilidad de hiper-mutación para saltar fuera de óptimos locales
            escala = 3.0 if random.random() < 0.15 else 1.0
            mutado[i, 0] = np.clip(mutado[i, 0] + random.gauss(0, 0.0055 * escala), RADIO_ALAMBRE * 3.0, RADIO_MAX)
            mutado[i, 1] = np.clip(mutado[i, 1] + random.gauss(0, 0.30 * escala), -0.35, 1.65)
            mutado[i, 2] = np.clip(mutado[i, 2] + random.gauss(0, 0.0035 * escala), 0.0035, dz_max_paso)
    return mutado

def ejecutar_optimizacion():
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    print("=" * 104)
    print(" OPTIMIZACIÓN MONOBANDA EXCLUSIVA PARA 5.50 GHz (2.44 GHz IGNORADA EN EL ALGORITMO)")
    print(f" Cilindro: R={RADIO_MAX*100:.1f} cm | H={ALTURA_MAX*100:.1f} cm | Ventana Longitud: {LONGITUD_MIN_M*100:.1f} a {LONGITUD_MAX_M*100:.1f} cm")
    print(f" Variación aumentada: Mutación={TASA_MUTACION*100:.0f}% | Inmigración aleatoria=10% por generación")
    print("=" * 104)

    for gen in range(GENERACIONES):
        evaluados = []
        for ind in poblacion:
            fit, s55, z55, sp3d, rip_xy, aire_mm, len_m = evaluar_individuo(ind)
            evaluados.append((fit, s55, z55, sp3d, rip_xy, aire_mm, len_m, ind))

        evaluados.sort(key=lambda x: x[0], reverse=True)

        if gen % 32 == 0 or gen == GENERACIONES - 1:
            bfit, bs55, bz55, bsp, brip, baire, blen, mejor_ind = evaluados[0]
            info_nec = ""
            if PYNEC_DISPONIBLE and baire > 1.0:
                nodos_g = decodificar_alambre_a_cartesianas(mejor_ind)
                res_nec55 = simular_un_alambre_pynec(nodos_g, FREQ_55)
                if res_nec55 is not None:
                    _, _, gmax55, gmin55, _, sd55 = res_nec55
                    info_nec = f" | NEC_5.5G:[{gmin55:4.1f} a {gmax55:4.1f}]dBi (SD:{sd55:.2f})"

            print(f"Gen {gen:4d}/{GENERACIONES} | Fit: {bfit:7.2f} | "
                  f"S11(5.5G): {bs55:5.1f} dB | "
                  f"Spread_3D: {bsp:4.2f} dB | Rizado_XY: {brip:4.2f} dB | L: {blen*100:5.2f} cm{info_nec}")

        mejores_elite = [copiar_individuo(item[7]) for item in evaluados[:ELITE_SIZE]]
        nueva_poblacion = [copiar_individuo(elite) for elite in mejores_elite]

        # Inyección de 10% de sangre nueva aleatoria para impedir estancamientos
        num_inmigrantes = TAMANO_POBLACION // 10
        for _ in range(num_inmigrantes):
            nueva_poblacion.append(crear_individuo())

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1, padre2 = random.sample(mejores_elite, 2)
            hijo = cruzar_individuos(padre1, padre2)
            nueva_poblacion.append(mutar_individuo(hijo))

        poblacion = nueva_poblacion

    bfit, bs55, bz55, bsp, brip, baire, blen, campeon = evaluados[0]
    nodos_m = decodificar_alambre_a_cartesianas(campeon)
    nodos_cm = nodos_m * 100.0

    nombre_archivo_nec = "antena_55ghz_pura_esp32c5.nec"
    contenido_nec = generar_archivo_nec_campeon(nodos_m, filename=nombre_archivo_nec)

    print("\n" + "=" * 104)
    print(" ¡ANTENA OPTIMIZADA 100% PARA 5.50 GHz COMPLETADA!")
    print(f" * S11 @ 5.50 GHz (Objetivo)     : {bs55:6.2f} dB (Z_in = {bz55.real:5.2f} {bz55.imag:+6.2f}j Ohm)")
    if PYNEC_DISPONIBLE:
        n55 = simular_un_alambre_pynec(nodos_m, FREQ_55)
        if n55:
            print(f" * PyNEC MoM @ 5.50 GHz          : Ganancia = [{n55[3]:5.2f} a {n55[2]:+5.2f}] dBi (SD={n55[5]:.2f} dB)")
    print(f" * Varianza Global 3D (5.50 GHz) : {bsp:5.2f} dB")
    print(f" * Rizado Azimutal X-Y (5.50 GHz): {brip:5.2f} dB (Dona redonda sin abolladuras)")
    print(f" * Longitud total desplegada     : {blen*100:.2f} cm")
    print(f" * Aire libre auto-intersección  : {baire:.2f} mm")
    print(f" * Archivo NEC2 guardado         : {nombre_archivo_nec}")
    print("=" * 104)

    print(f"\n* --- ARCHIVO NEC2 DEL CAMPEÓN ({nombre_archivo_nec}) ---")
    print(contenido_nec)

    print("\n// --- COPIAR Y PEGAR EN OPENSCAD (UN SOLO ALAMBRE 5.50 GHz) ---")
    print(f"radio_alambre_mm = {RADIO_ALAMBRE * 1000.0:.2f};")
    print("nodos_alambre = [")
    for idx, pt in enumerate(nodos_cm):
        coma = "," if idx < len(nodos_cm) - 1 else ""
        print(f"    [{pt[0]:7.3f}, {pt[1]:7.3f}, {pt[2]:7.3f}]{coma} // Paso {idx}")
    print("];")

if __name__ == "__main__":
    ejecutar_optimizacion()
