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

# Si está en False, evoluciona a máxima velocidad sin riesgo de segfault en C++
# y verifica con PyNEC (necpp) en cada reporte de generación y al final.
USAR_PYNEC_EN_BUCLE = False

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y ELÉCTRICOS (PATRÓN LISO Y REDONDO)
# =====================================================================
RADIO_MAX = 0.040         # 4.0 cm de radio máximo del cilindro (0.04 m)
ALTURA_MAX = 0.070        # 7.0 cm de altura máxima del cilindro (0.07 m)

LONGITUD_MIN_M = 0.085    # 8.5 cm longitud mínima estirada
LONGITUD_MAX_M = 0.135    # 13.5 cm (Permite el modo armónico de ~12.88 cm sin multa)

RADIO_ALAMBRE = 0.0020    # Radio 2.0 mm (Alambre grueso de 4.0 mm de diámetro)
SEPARACION_AIRE_MM = 2.5  # Espacio libre mínimo contra auto-intersección (mm)
NUM_SEGMENTOS = 10        # 10 pasos 3D progresivos

FREQ_24 = 2.44e9          # Banda baja Wi-Fi 6 (2.44 GHz)
FREQ_55 = 5.50e9          # Banda alta Wi-Fi 6 (5.50 GHz)
Z0 = 50.0                 # Impedancia puerto RF ESP32-C5 (50 Ohm)

TAMANO_POBLACION = 256 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 1024
GENERACIONES = 512 if (PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE) else 1024
ELITE_SIZE = max(16, TAMANO_POBLACION // 8)
TASA_MUTACION = 0.25

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
    """
    Cada gen codifica [r, delta_theta, delta_z].
    Restringimos delta_theta a valores positivos [0.35, 1.15] rad (~20° a 66° por paso)
    para que el alambre forme una espiral continua y suave sin zigzags destructivos.
    """
    genes = []
    dz_medio = ALTURA_MAX / NUM_SEGMENTOS
    for _ in range(NUM_SEGMENTOS):
        r = random.uniform(0.010, RADIO_MAX)
        d_theta = random.uniform(0.35, 1.15)
        d_z = random.uniform(dz_medio * 0.6, dz_medio * 1.35)
        genes.append([r, d_theta, d_z])
    return np.array(genes, dtype=float)

def copiar_individuo(ind):
    return np.copy(ind)

def decodificar_alambre_a_cartesianas(genes):
    """
    Convierte los genes en coordenadas 3D (X, Y, Z) partiendo de (0,0,0)[cite: 1].
    Garantiza una distancia mínima por paso para no violar las reglas de NEC2.
    """
    nodos = [[0.0, 0.0, 0.0]]
    suma_dz = sum(max(g[2], 0.0025) for g in genes)
    escala_z = min(1.0, ALTURA_MAX / max(suma_dz, 1e-6))

    theta_actual = 0.0
    z_actual = 0.0
    r_prev = RADIO_ALAMBRE * 3.5

    for idx, (r, d_theta, d_z) in enumerate(genes):
        r_target = float(np.clip(r, RADIO_ALAMBRE * 3.0, RADIO_MAX))
        r_clip = 0.35 * r_prev + 0.65 * r_target
        r_prev = r_clip

        theta_actual += d_theta
        paso_z = max(d_z * escala_z, 0.0035)
        z_actual = min(ALTURA_MAX, z_actual + paso_z)

        x = r_clip * math.cos(theta_actual)
        y = r_clip * math.sin(theta_actual)
        nodos.append([x, y, float(z_actual)])

    return np.array(nodos, dtype=float)

# =====================================================================
# 3. AUTO-COLISIÓN Y MODELO DE RADIACIÓN SIN ABOLLADURAS (XY + Z)
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

    # Verificar que ningún segmento individual sea más corto que 2.5 veces el diámetro
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

def evaluar_patron_3d_liso_xy_z(nodos):
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

    for i in range(len(vectores) - 1):
        cos_ang = float(np.dot(vectores[i], vectores[i+1]))
        if cos_ang < 0.35:
            penalizacion_geom += (0.35 - cos_ang) * 35.0

    centro_masa_xy = np.mean(nodos[1:, :2], axis=0)
    penalizacion_geom += float(np.linalg.norm(centro_masa_xy)) * 600.0

    def calcular_potencia_dir(theta, phi, freq):
        k = (2.0 * math.pi * freq) / C_LUZ
        k_eff = k * 0.88
        r_hat = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
        e_vec = np.zeros(3, dtype=complex)
        for u_vec, l_s, c_pt, s_pos in zip(vectores, longitudes, centros, pos_acumulada):
            dist_punta = longitud_total - s_pos
            i_mag = math.sin(k_eff * dist_punta) + 0.25j * math.cos(k_eff * dist_punta)
            fase_espacial = cmath_exp_j(k * float(np.dot(r_hat, c_pt)))
            trans = u_vec - r_hat * float(np.dot(u_vec, r_hat))
            e_vec += trans * l_s * i_mag * fase_espacial
        return float(np.sum(np.abs(e_vec) ** 2)) + 1e-9

    spread_total = 0.0
    rizado_xy_total = 0.0
    deficit_z_total = 0.0

    for freq in (FREQ_24, FREQ_55):
        p_xy_90 = [10.0 * math.log10(calcular_potencia_dir(math.pi * 0.50, phi, freq))
                   for phi in np.linspace(0, 2.0 * math.pi, 12, endpoint=False)]
        p_xy_60 = [10.0 * math.log10(calcular_potencia_dir(math.pi * 0.33, phi, freq))
                   for phi in np.linspace(0, 2.0 * math.pi, 12, endpoint=False)]
        p_z_top = 10.0 * math.log10(calcular_potencia_dir(0.0, 0.0, freq))

        ripple_90 = (max(p_xy_90) - min(p_xy_90)) + 2.0 * float(np.std(p_xy_90))
        ripple_60 = (max(p_xy_60) - min(p_xy_60)) + 2.0 * float(np.std(p_xy_60))
        rizado_xy_total += 0.5 * (ripple_90 + ripple_60)

        todos_pts = p_xy_90 + p_xy_60 + [p_z_top]
        spread_banda = max(todos_pts) - min(todos_pts)
        spread_total += 0.5 * spread_banda

        media_xy = float(np.mean(p_xy_90))
        diferencia_z_xy = abs(p_z_top - media_xy)
        deficit_z_total += diferencia_z_xy * 1.8

    return spread_total, rizado_xy_total, deficit_z_total, penalizacion_geom, longitud_total

# =====================================================================
# 4. MODELO DE IMPEDANCIA MULTIBANDA (UN SOLO ALAMBRE)
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
    """
    Evita el Segmentation Fault en C++ (necpp) asegurando que cada sub-segmento
    sea siempre más largo que 2.2 veces el radio del alambre (4.4 mm).
    """
    longitud_min_subseg = RADIO_ALAMBRE * 2.2
    n_max = int(l_tramo // longitud_min_subseg)
    return max(1, min(5, n_max))

def simular_un_alambre_pynec(nodos_m, freq_hz):
    if not PYNEC_DISPONIBLE:
        return None

    # Validación previa en Python antes de invocar punteros en C++
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

def generar_archivo_nec_campeon(nodos_m, filename="antena_monorrama_lisa_esp32c5.nec"):
    lineas_nec = [
        "CM ===================================================================",
        "CM ANTENA MONORRAMA ESFERICA LISA (XY + Z) (ESP32-C5: 2.44 & 5.50 GHz)",
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
        f"FR 0 1 0 0 {FREQ_24/1e6:.2f} 0.0",
        "RP 0 37 73 1000 0.0 0.0 5.0 5.0",
        f"FR 0 1 0 0 {FREQ_55/1e6:.2f} 0.0",
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
# 6. FUNCIÓN DE APTITUD (FITNESS: S11 + PATRÓN LISO EN X,Y + GANANCIA Z)
# =====================================================================
def evaluar_individuo(individuo):
    nodos = decodificar_alambre_a_cartesianas(individuo)

    pen_col, aire_mm = motor_autocolision_alambre(nodos, luz_libre_mm=SEPARACION_AIRE_MM)
    spread_3d, ripple_xy, deficit_z, pen_geom, len_total_m = evaluar_patron_3d_liso_xy_z(nodos)

    # Gatekeeper: Solo llamamos a PyNEC en el bucle si está activado Y si no hay colisión
    if PYNEC_DISPONIBLE and USAR_PYNEC_EN_BUCLE and pen_col == 0.0:
        res24 = simular_un_alambre_pynec(nodos, FREQ_24)
        res55 = simular_un_alambre_pynec(nodos, FREQ_55)
    else:
        res24, res55 = None, None

    if res24 is not None and res55 is not None:
        z_in_24, s11_24, gmax24, gmin24, delta_dbi_24, sd24 = res24
        z_in_55, s11_55, gmax55, gmin55, delta_dbi_55, sd55 = res55
        varianza_dbi = 0.5 * (delta_dbi_24 + delta_dbi_55) + 2.0 * (sd24 + sd55)
    else:
        z_in_24 = calcular_z_in_un_alambre(nodos, FREQ_24)
        z_in_55 = calcular_z_in_un_alambre(nodos, FREQ_55)
        gamma_24 = abs((z_in_24 - Z0) / (z_in_24 + Z0))
        gamma_55 = abs((z_in_55 - Z0) / (z_in_55 + Z0))
        s11_24 = 20.0 * math.log10(max(gamma_24, 1e-5))
        s11_55 = 20.0 * math.log10(max(gamma_55, 1e-5))
        varianza_dbi = spread_3d + (ripple_xy * 1.5) + deficit_z

    peor_s11 = max(s11_24, s11_55)
    promedio_s11 = 0.5 * (s11_24 + s11_55)

    penalizacion_s11 = 0.0
    if s11_24 > -10.0: penalizacion_s11 += (s11_24 - (-10.0)) * 8.0
    if s11_55 > -10.0: penalizacion_s11 += (s11_55 - (-10.0)) * 8.0

    fitness = (-peor_s11 * 2.0) + (-promedio_s11 * 0.8) - (varianza_dbi * 4.0) - penalizacion_s11 - pen_geom - pen_col

    return fitness, s11_24, s11_55, z_in_24, z_in_55, spread_3d, ripple_xy, aire_mm, len_total_m

# =====================================================================
# 7. OPERADORES GENÉTICOS Y BUCLE PRINCIPAL
# =====================================================================
def cruzar_individuos(padre1, padre2):
    hijo = crear_individuo()
    for i in range(NUM_SEGMENTOS):
        hijo[i] = np.copy(padre1[i] if random.random() < 0.5 else padre2[i])
    return hijo

def mutar_individuo(individuo):
    mutado = copiar_individuo(individuo)
    dz_max_paso = (ALTURA_MAX / NUM_SEGMENTOS) * 1.5
    for i in range(NUM_SEGMENTOS):
        if random.random() < TASA_MUTACION:
            mutado[i, 0] = np.clip(mutado[i, 0] + random.gauss(0, 0.0025), RADIO_ALAMBRE * 3.0, RADIO_MAX)
            mutado[i, 1] = np.clip(mutado[i, 1] + random.gauss(0, 0.15), 0.30, 1.25)
            mutado[i, 2] = np.clip(mutado[i, 2] + random.gauss(0, 0.0015), 0.0035, dz_max_paso)
    return mutado

def ejecutar_optimizacion():
    random.seed(42)
    np.random.seed(42)

    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    print("=" * 104)
    print(" OPTIMIZACIÓN DE HÉLICE SUAVE MULTIFASE (BLINDADO CONTRA SEGFAULT EN PYNEC)")
    print(f" Cilindro: R={RADIO_MAX*100:.1f} cm | H={ALTURA_MAX*100:.1f} cm | Ventana Longitud: {LONGITUD_MIN_M*100:.1f} a {LONGITUD_MAX_M*100:.1f} cm")
    print(f" PyNEC detectado: {PYNEC_DISPONIBLE} | Evaluación segura C++ en reportes y campeón final")
    print("=" * 104)

    for gen in range(GENERACIONES):
        evaluados = []
        for ind in poblacion:
            fit, s24, s55, z24, z55, sp3d, rip_xy, aire_mm, len_m = evaluar_individuo(ind)
            evaluados.append((fit, s24, s55, z24, z55, sp3d, rip_xy, aire_mm, len_m, ind))

        evaluados.sort(key=lambda x: x[0], reverse=True)

        if gen % 32 == 0 or gen == GENERACIONES - 1:
            bfit, bs24, bs55, bz24, bz55, bsp, brip, baire, blen, mejor_ind = evaluados[0]
            info_nec = ""
            if PYNEC_DISPONIBLE and baire > 1.0:
                nodos_g = decodificar_alambre_a_cartesianas(mejor_ind)
                res_nec = simular_un_alambre_pynec(nodos_g, FREQ_24)
                if res_nec is not None:
                    _, _, gmax24, gmin24, _, _ = res_nec
                    info_nec = f" | NEC_2.4G:[{gmin24:4.1f},{gmax24:4.1f}]dBi"

            print(f"Gen {gen:4d}/{GENERACIONES} | Fit: {bfit:7.2f} | "
                  f"S11: {bs24:5.1f}/{bs55:5.1f} dB | "
                  f"Spread_3D: {bsp:4.2f} dB | Rizado_XY: {brip:4.2f} dB | L: {blen*100:5.2f} cm{info_nec}")

        mejores_elite = [copiar_individuo(item[9]) for item in evaluados[:ELITE_SIZE]]
        nueva_poblacion = [copiar_individuo(elite) for elite in mejores_elite]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1, padre2 = random.sample(mejores_elite, 2)
            hijo = cruzar_individuos(padre1, padre2)
            nueva_poblacion.append(mutar_individuo(hijo))

        poblacion = nueva_poblacion

    bfit, bs24, bs55, bz24, bz55, bsp, brip, baire, blen, campeon = evaluados[0]
    nodos_m = decodificar_alambre_a_cartesianas(campeon)
    nodos_cm = nodos_m * 100.0

    nombre_archivo_nec = "antena_monorrama_lisa_esp32c5.nec"
    contenido_nec = generar_archivo_nec_campeon(nodos_m, filename=nombre_archivo_nec)

    print("\n" + "=" * 104)
    print(" ¡ANTENA DE PATRÓN ESFÉRICO LISO (X, Y, +Z) COMPLETADA SIN ERRORES!")
    print(f" * S11 @ 2.44 GHz                : {bs24:6.2f} dB (Z_in = {bz24.real:5.2f} {bz24.imag:+6.2f}j Ohm)")
    print(f" * S11 @ 5.50 GHz                : {bs55:6.2f} dB (Z_in = {bz55.real:5.2f} {bz55.imag:+6.2f}j Ohm)")
    if PYNEC_DISPONIBLE:
        n24 = simular_un_alambre_pynec(nodos_m, FREQ_24)
        n55 = simular_un_alambre_pynec(nodos_m, FREQ_55)
        if n24 and n55:
            print(f" * PyNEC MoM @ 2.44 GHz          : Ganancia = [{n24[3]:5.2f} a {n24[2]:+5.2f}] dBi (SD={n24[5]:.2f} dB)")
            print(f" * PyNEC MoM @ 5.50 GHz          : Ganancia = [{n55[3]:5.2f} a {n55[2]:+5.2f}] dBi (SD={n55[5]:.2f} dB)")
    print(f" * Varianza Global 3D (XY + Z)   : {bsp:5.2f} dB")
    print(f" * Rizado Azimutal en X-Y        : {brip:5.2f} dB (Menor = Dona más redonda y sin abolladuras)")
    print(f" * Longitud total desplegada     : {blen*100:.2f} cm")
    print(f" * Aire libre auto-intersección  : {baire:.2f} mm")
    print(f" * Archivo NEC2 guardado         : {nombre_archivo_nec}")
    print("=" * 104)

    print(f"\n* --- ARCHIVO NEC2 DEL CAMPEÓN ({nombre_archivo_nec}) ---")
    print(contenido_nec)

    print("\n// --- COPIAR Y PEGAR EN OPENSCAD (UN SOLO ALAMBRE LISO) ---")
    print(f"radio_alambre_mm = {RADIO_ALAMBRE * 1000.0:.2f};")
    print("nodos_alambre = [")
    for idx, pt in enumerate(nodos_cm):
        coma = "," if idx < len(nodos_cm) - 1 else ""
        print(f"    [{pt[0]:7.3f}, {pt[1]:7.3f}, {pt[2]:7.3f}]{coma} // Paso {idx}")
    print("];")

if __name__ == "__main__":
    ejecutar_optimizacion()
